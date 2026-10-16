"""MOSS-Image DiT LoRA：只作用于 DiT，不修改 Qwen 文本编码器。

以 forward hook 方式叠加 ``scale * up(down(x))``，不改动原权重，因此：
- 卸载 LoRA 后输出与原模型逐位一致；
- 与组件 / 顺序 offload 兼容（hook 跟随输入所在设备）。

支持的键名格式（DiT 权重键与 diffusers / ComfyUI 导出一致，如 ``layers.0.proj.q``）：
- diffusers / PEFT：``transformer.layers.0.proj.q.lora_A.weight`` / ``lora_B.weight``
- ComfyUI：``diffusion_model.layers.0.proj.q.lora_A.weight`` 或 ``lora_down`` / ``lora_up``
- kohya：``lora_unet_layers_0_proj_q.lora_down.weight`` / ``lora_up.weight`` / ``alpha``
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

_PREFIXES = (
    "base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.",
    "lora_unet_", "lora_transformer_", "unet.",
)
_TEXT_ENCODER_PREFIXES = ("lora_te", "text_encoder", "te_", "text_model", "lora_te1_", "lora_te2_")
_DOWN = (".lora_A.weight", ".lora_down.weight", ".lora.down.weight", ".lora_A.default.weight")
_UP = (".lora_B.weight", ".lora_up.weight", ".lora.up.weight", ".lora_B.default.weight")


def _split(key):
    for kind, suffixes in (("down", _DOWN), ("up", _UP), ("alpha", (".alpha",))):
        for suffix in suffixes:
            if key.endswith(suffix):
                return key[: -len(suffix)], kind
    return None, None


def _strip(name):
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if name.startswith(prefix):
                name, changed = name[len(prefix):], True
    return name


def parse_lora(state_dict, transformer):
    """把 LoRA 键映射到 DiT 的 nn.Linear；返回 {模块名: (down, up, alpha)} 与跳过信息。"""
    linears = {
        name: module for name, module in transformer.named_modules()
        if isinstance(module, nn.Linear) or getattr(module, "is_gguf_linear", False)
    }
    kohya = {name.replace(".", "_"): name for name in linears}
    groups, skipped_te, unknown = {}, 0, []
    for key, tensor in state_dict.items():
        base, kind = _split(key)
        if base is None:
            unknown.append(key)
            continue
        if base.lower().startswith(_TEXT_ENCODER_PREFIXES):
            # 官方要求文本编码器保持冻结，这里直接忽略 TE 部分。
            skipped_te += 1
            continue
        name = _strip(base)
        name = name if name in linears else kohya.get(name, name)
        groups.setdefault(name, {})[kind] = tensor
    weights, unmatched = {}, []
    for name, parts in groups.items():
        if name not in linears or "down" not in parts or "up" not in parts:
            unmatched.append(name)
            continue
        down, up = parts["down"], parts["up"]
        linear = linears[name]
        if down.ndim != 2 or up.ndim != 2 or down.shape[1] != linear.in_features \
                or up.shape[0] != linear.out_features or up.shape[1] != down.shape[0]:
            raise ValueError(f"LoRA 形状与 DiT 不匹配：{name} down{tuple(down.shape)} up{tuple(up.shape)}")
        alpha = float(parts["alpha"]) if "alpha" in parts else float(down.shape[0])
        weights[name] = (down, up, alpha)
    return weights, skipped_te, unmatched + unknown


class _LoRAHook:
    def __init__(self):
        self.entries = []  # [(down, up, scale)]，按设备缓存副本
        self.cache = {}

    def __call__(self, module, inputs, output):
        x = inputs[0]
        key = (x.device, x.dtype)
        if key not in self.cache:
            self.cache[key] = [
                (down.to(x.device, x.dtype), up.to(x.device, x.dtype), scale)
                for down, up, scale in self.entries
            ]
        for down, up, scale in self.cache[key]:
            output = output + F.linear(F.linear(x, down), up) * scale
        return output


class LoRAManager:
    """管理当前挂在 DiT 上的 LoRA 组合；组合不变时不重复加载。"""

    def __init__(self):
        self.key = ()
        self.handles = []
        self.file_cache = {}

    def _load(self, path):
        path = str(Path(path).resolve())
        stat = Path(path).stat()
        cache_key = (path, stat.st_size, stat.st_mtime)
        if cache_key not in self.file_cache:
            if len(self.file_cache) >= 4:
                self.file_cache.pop(next(iter(self.file_cache)))
            self.file_cache[cache_key] = load_file(path, device="cpu")
        return self.file_cache[cache_key]

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.key = ()

    def apply(self, transformer, loras):
        """loras: [(名称, 文件路径, 权重)]；返回每个 LoRA 的匹配统计。"""
        key = tuple((name, str(path), float(weight)) for name, path, weight in loras)
        if key == self.key:
            return []
        self.remove()
        hooks, report = {}, []
        linears = dict(transformer.named_modules())
        for name, path, weight in loras:
            if float(weight) == 0:
                report.append((name, 0, 0, 0))
                continue
            weights, skipped_te, unmatched = parse_lora(self._load(path), transformer)
            if not weights:
                raise ValueError(
                    f"LoRA「{name}」没有任何键能匹配 MOSS-Image DiT（可能是其他模型的 LoRA）。"
                )
            for module_name, (down, up, alpha) in weights.items():
                scale = float(weight) * alpha / down.shape[0]
                hooks.setdefault(module_name, _LoRAHook()).entries.append(
                    (down.to(torch.bfloat16), up.to(torch.bfloat16), scale)
                )
            report.append((name, len(weights), skipped_te, len(unmatched)))
        for module_name, hook in hooks.items():
            self.handles.append(linears[module_name].register_forward_hook(hook))
        self.key = key
        return report
