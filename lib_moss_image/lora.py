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
    # .diff / .diff_b：完整的权重 / bias 增量（Kijai 等工具提取的 LoRA 会给 norm 层带上）
    for kind, suffixes in (("down", _DOWN), ("up", _UP), ("alpha", (".alpha",)),
                           ("diff_b", (".diff_b",)), ("diff", (".diff",))):
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
    """把 LoRA 键映射到 DiT；返回 ({模块名: (down, up, alpha)}, {参数名: 增量}, 跳过的 TE 键数, 未匹配列表)。"""
    linears = {
        name: module for name, module in transformer.named_modules()
        if isinstance(module, nn.Linear) or getattr(module, "is_gguf_linear", False)
    }
    modules = dict(transformer.named_modules())
    kohya = {name.replace(".", "_"): name for name in modules if name}
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
        name = name if name in modules or name.endswith(".proj.qkv") else kohya.get(name, name)
        groups.setdefault(name, {})[kind] = tensor
    weights, diffs, unmatched = {}, {}, []
    for name, parts in groups.items():
        for kind, suffix in (("diff", ".weight"), ("diff_b", ".bias")):
            if kind in parts:
                diffs[name + suffix] = parts[kind]
        if "down" not in parts or "up" not in parts:
            if not ({"diff", "diff_b"} & parts.keys()):
                unmatched.append(name)
            continue
        down, up = parts["down"], parts["up"]
        alpha = float(parts["alpha"]) if "alpha" in parts else float(down.shape[0])
        # ComfyUI 原生格式的 LoRA 训练在合并的 qkv 上：up 按行拆成 q / k / v，down 三者共用（数学上等价）
        targets = [name]
        if name not in linears and name.endswith(".proj.qkv"):
            base = name[: -len("qkv")]
            targets = [base + c for c in "qkv"]
            ups = up.chunk(3, 0)
        else:
            ups = [up]
        for target, part_up in zip(targets, ups):
            if target not in linears:
                unmatched.append(name)
                break
            linear = linears[target]
            if down.ndim != 2 or part_up.ndim != 2 or down.shape[1] != linear.in_features \
                    or part_up.shape[0] != linear.out_features or part_up.shape[1] != down.shape[0]:
                raise ValueError(f"LoRA 形状与 DiT 不匹配：{name} down{tuple(down.shape)} up{tuple(up.shape)}")
            weights[target] = (down, part_up.contiguous(), alpha)
    params = dict(transformer.named_parameters())
    for pname in list(diffs):
        if pname not in params or tuple(diffs[pname].shape) != tuple(params[pname].shape):
            unmatched.append(pname)
            diffs.pop(pname)
    return weights, diffs, skipped_te, unmatched + unknown


def _param_storage(transformer, pname):
    """参数实际存放的张量：「逐层搬运」时参数是 meta，真实数据在 accelerate hook 的 weights_map 里"""
    module_name, attr = pname.rsplit(".", 1) if "." in pname else ("", pname)
    module = transformer.get_submodule(module_name) if module_name else transformer
    tensor = getattr(module, attr)
    if tensor.device.type == "meta":
        hook = getattr(module, "_hf_hook", None)
        weights_map = getattr(hook, "weights_map", None)
        if weights_map is None:
            raise RuntimeError(f"找不到 {pname} 的权重存放位置")
        return weights_map[attr]
    return tensor.data


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
        self.backups = []  # [(transformer, 参数名, 原值)]：.diff 增量直接改权重，卸载时原样恢复

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
        # 按参数名找当前存放位置再恢复：offload 期间参数会在 CPU / 显卡之间换新的张量
        for transformer, pname, original in reversed(self.backups):
            _param_storage(transformer, pname).copy_(original)
        self.handles = []
        self.backups = []
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
            weights, diffs, skipped_te, unmatched = parse_lora(self._load(path), transformer)
            if not weights and not diffs:
                raise ValueError(
                    f"LoRA「{name}」没有任何键能匹配 MOSS-Image DiT（可能是其他模型的 LoRA）。"
                )
            for module_name, (down, up, alpha) in weights.items():
                scale = float(weight) * alpha / down.shape[0]
                hooks.setdefault(module_name, _LoRAHook()).entries.append(
                    (down.to(torch.bfloat16), up.to(torch.bfloat16), scale)
                )
            for pname, delta in diffs.items():
                storage = _param_storage(transformer, pname)
                self.backups.append((transformer, pname, storage.detach().to("cpu", copy=True)))
                storage.add_(delta.to(storage.device, storage.dtype) * float(weight))
            report.append((name, len(weights) + len(diffs), skipped_te, len(unmatched)))
        for module_name, hook in hooks.items():
            self.handles.append(linears[module_name].register_forward_hook(hook))
        self.key = key
        return report
