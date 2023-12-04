# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import re
from safetensors.torch import load_file

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs

@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf"),
    model_name: Optional[str] = None,
    inner_k_tiles: int = 8,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    pt_model_map_json = checkpoint_dir / "pytorch_model.bin.index.json"
    st_model_map_json = checkpoint_dir / "model.safetensors.index.json"
    pt_model = checkpoint_dir / "pytorch_model.bin"
    st_model = checkpoint_dir / "model.safetensors"
    if pt_model_map_json.is_file() or st_model_map_json.is_file():
        model_map_json = pt_model_map_json if pt_model_map_json.is_file() else st_model_map_json
        with open(model_map_json) as json_map:
            bin_index = json.load(json_map)
        bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    elif pt_model.is_file():
        bin_files = {pt_model}
    elif st_model.is_file():
        bin_files = {st_model}
    else:
        print("cannot find model weights")
        return

    weight_map = {
        "model.embed_tokens": "tok_embeddings",
        "model.layers.{}.self_attn.q_proj": "layers.{}.attention.wq",
        "model.layers.{}.self_attn.k_proj": "layers.{}.attention.wk",
        "model.layers.{}.self_attn.v_proj": "layers.{}.attention.wv",
        "model.layers.{}.self_attn.o_proj": "layers.{}.attention.wo",
        'model.layers.{}.self_attn.rotary_emb': None,
        'model.layers.{}.mlp.gate_proj': 'layers.{}.feed_forward.w1',
        "model.layers.{}.mlp.up_proj": "layers.{}.feed_forward.w3",
        "model.layers.{}.mlp.down_proj": "layers.{}.feed_forward.w2",
        "model.layers.{}.input_layernorm": "layers.{}.attention_norm",
        "model.layers.{}.post_attention_layernorm": "layers.{}.ffn_norm",
        "model.norm": "norm",
        "lm_head": "output",
    }

    def permute(w, n_head, dim=0):
        return (
            w.view(*w.shape[:dim], n_head, 2, config.head_dim // 2, *w.shape[dim + 1:])
            .transpose(dim + 1, dim + 2)
            .reshape(*w.shape)
        )

    merged_result = {}
    for file in sorted(bin_files):
        if str(file).endswith(".bin"):
            state_dict = torch.load(str(file), map_location="cpu", mmap=True, weights_only=True)
        else:
            state_dict = load_file(str(file), device="cpu")
        merged_result.update(state_dict)

    final_result = {}
    for key, value in merged_result.items():
        if "layers" in key:
            abstract_key = re.sub(r'(\d+)', '{}', key)
            abstract_layer, suffix = abstract_key.rsplit(".", 1)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map[abstract_layer]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num) + f".{suffix}"
        else:
            layer, suffix = key.rsplit(".", 1)
            new_key = weight_map[layer] + f".{suffix}"

        final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "bias" in key:
            del final_result[key]
        if "qweight" in key:
            wf = torch.tensor(list(range(0, 32, 4)), dtype=torch.int32).unsqueeze(0)
            weight_int32 = torch.bitwise_right_shift(
                torch.unsqueeze(final_result[key], 1).expand(-1, 8, -1),
                torch.tensor(list(range(0, 32, 4)), dtype=torch.int32).unsqueeze(0).unsqueeze(-1),
            ).bitwise_and(15)
            weight_int32 = weight_int32.reshape(-1, weight_int32.shape[2]).transpose(0, 1).contiguous()
            g_idx = final_result[key.replace("qweight", "g_idx")]
            rev_g_idx = torch.argsort(g_idx)
            final_result[key] = weight_int32[:, rev_g_idx]
            final_result[key.replace("qweight", "g_idx")] = rev_g_idx
        if "scales" in key:
            scales = final_result[key]
            qzeros = final_result[key.replace("scales", "qzeros")]
            qzeros = torch.bitwise_right_shift(
                torch.unsqueeze(qzeros, 2).expand(-1, -1, 8),
                torch.tensor(list(range(0, 32, 4)), dtype=torch.int32).unsqueeze(0).unsqueeze(0),
            ).bitwise_and(15)
            qzeros = qzeros + 1
            zeros = (8 - qzeros.reshape(scales.shape)) * scales
            scales_and_zeros = torch.cat(
                [
                    scales.reshape(scales.size(0), scales.size(1), 1),
                    zeros.reshape(zeros.size(0), zeros.size(1), 1),
                ],
                2,
            )
            final_result[key.replace("scales", "scales_and_zeros")] = scales_and_zeros
            del final_result[key]
            del final_result[key.replace("scales", "qzeros")]

    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            if "g_idx" in key:
                final_result[key.replace("wq", "wqkv")] = q
            else:
                dim = 1 if "scales_and_zeros" in key else 0
                q = permute(q, config.n_head, dim)
                k = permute(k, config.n_local_heads, dim)
                final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v], dim=dim)
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]

    for key in tuple(final_result.keys()):
        if "qweight" in key:
            weight_int32 = final_result[key]
            weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(weight_int32.cuda(), inner_k_tiles).to('cpu')
            final_result[key.replace("qweight", "weight")] = weight_int4pack
            del final_result[key]

    hf_config = checkpoint_dir / "config.json"
    assert hf_config.is_file()
    hf_config = json.load(open(hf_config))
    if "quantization_config" in hf_config and hf_config["quantization_config"]["quant_method"] == "gptq":
        model_name = checkpoint_dir / f"model_int4.g{hf_config['quantization_config']['group_size']}.pth"
    else:
        model_name = checkpoint_dir / "model.pth"
    print(f"Saving checkpoint to {model_name}")
    torch.save(final_result, model_name)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path("checkpoints/meta-llama/llama-2-7b-chat-hf"))
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--inner_k_tiles', type=int, default=8)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        inner_k_tiles=args.inner_k_tiles
    )
