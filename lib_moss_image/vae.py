"""diffusers 0.37.1 的 FLUX.2 small decoder 兼容层。

0.38 相比 Forge 当前版本仅在 VAE 中新增 ``decoder_block_out_channels``。
这里只替换 decoder，其余编码器、BN、decode 顺序仍使用官方类。
"""
from __future__ import annotations

import torch
from diffusers import AutoencoderKLFlux2
from diffusers.models.autoencoders.vae import Decoder


class MOSSAutoencoderKLFlux2(AutoencoderKLFlux2):
    def __init__(self, *args, decoder_block_out_channels=None, **kwargs):
        super().__init__(*args, **kwargs)
        if decoder_block_out_channels is not None:
            self.decoder = Decoder(
                in_channels=self.config.latent_channels,
                out_channels=self.config.out_channels,
                up_block_types=self.config.up_block_types,
                block_out_channels=decoder_block_out_channels,
                layers_per_block=self.config.layers_per_block,
                norm_num_groups=self.config.norm_num_groups,
                act_fn=self.config.act_fn,
                mid_block_add_attention=self.config.mid_block_add_attention,
            )
        self.register_to_config(decoder_block_out_channels=decoder_block_out_channels)


def load_moss_vae(root):
    # 官方明确要求 VAE 和解码过程使用 fp32。
    return MOSSAutoencoderKLFlux2.from_pretrained(
        root, torch_dtype=torch.float32, local_files_only=True
    ).eval()


def load_moss_vae_file(path, config_root):
    """顶部「VAE / Text Encoder」选中的单文件 VAE（ComfyUI 打包的 flux2-vae / moss_image_vae_bf16 等）。
    键名与官方 diffusers 文件一致；结构取官方 vae/config.json，权重统一转 fp32。"""
    from safetensors.torch import load_file

    config = MOSSAutoencoderKLFlux2.load_config(config_root, local_files_only=True)
    vae = MOSSAutoencoderKLFlux2.from_config(config)
    state = {k: (v.float() if v.is_floating_point() else v) for k, v in load_file(str(path)).items()}
    try:
        vae.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"VAE「{path}」与 MOSS-Image 的 FLUX.2 VAE 结构不符：{str(error)[:300]}") from None
    return vae.to(torch.float32).eval()
