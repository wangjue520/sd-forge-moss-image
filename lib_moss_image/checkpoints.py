"""把 MOSS-Image 接入顶部 Checkpoint / 「VAE / Text Encoder」下拉框，沿用 Forge 的使用习惯。

- 官方 Base / SFT 目录注册为 Checkpoint「MOSS-Image-Base」「MOSS-Image-SFT」；
- MOSS-Image 目录下的量化 DiT（GGUF / ConvRot）也注册进来；
- 放在 checkpoints / diffusion_models 里的 MOSS DiT 单文件按文件头识别；
- 「VAE / Text Encoder」中选中的 Qwen3.5 单文件作为量化文本编码器。
moss 预设下 Checkpoint 只列 MOSS 模型，其他预设隐藏它们（PRESET_CHECKPOINTS）。
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

from modules import sd_models, shared

from .runtime import MODEL_NAMES, is_model_bundle, model_roots

PRESET = "moss"
_HEADER_CACHE = {}


class MossCheckpointInfo(sd_models.CheckpointInfo):
    """不对应单个权重文件的虚拟 Checkpoint（官方 diffusers 目录 / MOSS 目录下的量化 DiT）。"""

    def __init__(self, title, filename, bundle, dit):
        self.filename = str(filename)
        self.name = self.name_for_extra = self.model_name = self.title = self.short_title = title
        self.is_safetensors = False
        self.metadata = {}
        self.hash = self.sha256 = self.shorthash = None
        self.ids = [title]
        self.moss = (bundle, dit)

    def calculate_shorthash(self):
        return None


def _header_keys(path):
    """读取 safetensors / GGUF 的张量名（带缓存），用于识别模型类型。"""
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return set()
    key = (str(path), stat.st_size, stat.st_mtime)
    if key in _HEADER_CACHE:
        return _HEADER_CACHE[key]
    names = set()
    try:
        if path.suffix.lower() in (".safetensors", ".sft"):
            with open(path, "rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                if size < 64 * 1024 * 1024:
                    names = set(json.loads(handle.read(size))) - {"__metadata__"}
        elif path.suffix.lower() == ".gguf":
            from modules_forge.packages import gguf

            names = {str(t.name) for t in gguf.GGUFReader(str(path)).tensors}
    except Exception:
        names = set()
    _HEADER_CACHE[key] = names
    return names


def _strip(name):
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer.", "model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def is_moss_dit_file(path):
    keys = {_strip(k) for k in _header_keys(path)}
    return {"video_embedder.weight", "text_embedder.proj.weight", "video_head_modulation.weight"} <= keys


def is_moss_text_encoder_file(path):
    """Qwen3.5（MOSS-Image 文本编码器）：含 linear_attn（GatedDeltaNet）层的 language_model。"""
    keys = {k.removeprefix("model.") for k in _header_keys(path)}
    return "language_model.embed_tokens.weight" in keys and any(".linear_attn.in_proj_qkv." in k for k in keys)


def _guess_bundle(name):
    have = [m for m in MODEL_NAMES if any(is_model_bundle(root / m, ()) for root in model_roots())]
    lower = name.lower()
    for model in MODEL_NAMES:
        if model.lower() in lower and model in have:
            return model
    return "SFT" if "SFT" in have else (have[0] if have else "SFT")


def register_checkpoints():
    """CHECKPOINT_PROVIDERS：每次 Forge 刷新 Checkpoint 列表后调用。"""
    from .quant import list_quant_files

    seen = {os.path.abspath(info.filename) for info in sd_models.checkpoints_list.values()}
    for model in MODEL_NAMES:
        for root in model_roots():
            if is_model_bundle(root / model):
                MossCheckpointInfo(f"MOSS-Image-{model}", root / model, model, None).register()
                break
    for root in model_roots():
        for name in list_quant_files(root):
            path = root / name
            if os.path.abspath(path) not in seen and is_moss_dit_file(path):
                MossCheckpointInfo(name, path, _guess_bundle(name), path).register()


def is_moss_checkpoint(info):
    """PRESET_CHECKPOINTS["moss"]：这个 Checkpoint 是否属于 MOSS-Image。"""
    if isinstance(info, MossCheckpointInfo):
        return True
    filename = getattr(info, "filename", "")
    return str(filename).lower().endswith((".gguf", ".safetensors")) and is_moss_dit_file(filename)


def resolve(info):
    """Checkpoint -> (官方组件目录 Base/SFT, 量化 DiT 文件或 None)。"""
    if isinstance(info, MossCheckpointInfo):
        return info.moss
    return _guess_bundle(Path(info.filename).name), Path(info.filename)


def selected_checkpoint(p=None):
    """生成时使用的 MOSS Checkpoint：API override > 顶部下拉框 > moss 预设记住的 > 默认。"""
    candidates = []
    if p is not None:
        candidates.append((getattr(p, "override_settings", None) or {}).get("sd_model_checkpoint"))
    candidates += [shared.opts.sd_model_checkpoint, getattr(shared.opts, f"forge_checkpoint_{PRESET}", None)]
    for name in candidates:
        info = sd_models.get_closet_checkpoint_match(name) if name else None
        if info is not None and is_moss_checkpoint(info):
            return info
    return default_checkpoint()


def default_checkpoint():
    for title in ("MOSS-Image-SFT", "MOSS-Image-Base"):
        if title in sd_models.checkpoint_aliases:
            return sd_models.checkpoint_aliases[title]
    return next((info for info in sd_models.checkpoints_list.values() if is_moss_checkpoint(info)), None)


def is_moss_vae_file(path):
    """MOSS-Image 使用的 FLUX.2 VAE（带 BatchNorm 统计量 bn.*，与 SD / SDXL / Flux.1 VAE 区分）。"""
    keys = _header_keys(path)
    return {"bn.running_mean", "bn.running_var", "encoder.conv_in.weight", "decoder.conv_in.weight"} <= keys


def selected_modules(p=None):
    """顶部「VAE / Text Encoder」的选择 -> {"te": 路径或 None, "vae": 路径或 None, "ignored": [不适用的文件名]}。
    没选的部分自动使用官方 Base / SFT 目录中的原版。"""
    result = {"te": None, "vae": None, "ignored": []}
    overrides = (getattr(p, "override_settings", None) or {}) if p is not None else {}
    selected = overrides.get("forge_additional_modules", shared.opts.forge_additional_modules) or []
    for path in selected:
        if result["te"] is None and is_moss_text_encoder_file(path):
            result["te"] = Path(path)
        elif result["vae"] is None and is_moss_vae_file(path):
            result["vae"] = Path(path)
        else:
            result["ignored"].append(Path(path).name)
    return result


def describe(info, modules):
    if info is None:
        return "未找到 MOSS-Image 模型，请在下方「模型下载」下载官方 Base / SFT。"
    bundle, dit = resolve(info)
    te, vae = modules["te"], modules["vae"]
    dit_text = "官方原版主模型" if dit is None else f"压缩版主模型 `{Path(dit).name}`"
    te_text = "官方原版文本编码器" if te is None else f"文本编码器 `{te.name}`"
    vae_text = "官方原版 VAE" if vae is None else f"VAE `{vae.name}`"
    text = f"当前使用：**{info.name}**（{dit_text}，{te_text}，{vae_text}；未选的部分来自官方 {bundle}）"
    if modules["ignored"]:
        text += (
            f"\n\n⚠️ 顶部「VAE / Text Encoder」里的 {'、'.join(modules['ignored'])} 不是 MOSS-Image 能用的模型，已忽略"
            "（MOSS 只能用 Qwen3.5 文本编码器和 FLUX.2 VAE）。"
        )
    return text
