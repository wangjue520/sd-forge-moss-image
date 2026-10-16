"""给 Forge Neo 本体打上 MOSS-Image 需要的小补丁（幂等：已打过的部分自动跳过）。

改动内容（都很小，且只在 moss 预设下起作用）：
- modules_forge/presets.py：UI Preset 新增 moss（Euler / Simple / 50 步 / CFG 3 / Shift 5），
  以及三个扩展接口 PROCESS_HOOKS / CHECKPOINT_PROVIDERS / PRESET_CHECKPOINTS；
- modules/processing.py：process_images 开头把 moss 预设的生成交给扩展；
- modules/sd_models.py：刷新 Checkpoint 列表时让扩展注册 MOSS 模型；
- modules_forge/main_entry.py：Checkpoint 下拉框按当前预设过滤。
Forge 升级覆盖这些文件后，下次启动会自动重新打上。
"""
from __future__ import annotations

import re
from pathlib import Path

MARK = "# MOSS-Image (sd-forge-moss-image)"

PRESET_VALUES = {"SAMPLERS": '"Euler"', "SCHEDULERS": '"Simple"', "STEPS": "50", "CFG": "3.0", "SHIFT": "5.0"}

HOOKS = '''

# presets whose model is not a Forge checkpoint: an extension registers `fn(p) -> Processed`
# and process_images() hands the whole generation over to it (e.g. "moss" -> MOSS-Image)
PROCESS_HOOKS: dict = {}
'''

PROVIDERS = '''

# extensions add their own entries to the Checkpoint list: each `fn()` runs at the end of sd_models.list_models()
CHECKPOINT_PROVIDERS: list = []

# checkpoints owned by a preset: `fn(checkpoint_info) -> bool`; the dropdown of that preset only lists
# its own checkpoints, and every other preset hides them (e.g. "moss" -> MOSS-Image models)
PRESET_CHECKPOINTS: dict = {}


def preset_checkpoint_tiles(preset: str, use_short: bool = False) -> list[str]:
    from modules import sd_models

    def keep(info) -> bool:
        if (owns := PRESET_CHECKPOINTS.get(preset)) is not None:
            return owns(info)
        return not any(owns(info) for owns in PRESET_CHECKPOINTS.values())

    return [x.short_title if use_short else x.name for x in sd_models.checkpoints_list.values() if keep(x)]
'''


def _patch_presets(text):
    if not re.search(r"^\s+moss = \d+", text, re.M):
        enum = re.search(r"^class PresetArch\(Enum\):\n((?:    \w+ = \d+.*\n|\s*#.*\n)+)", text, re.M)
        if enum is None:
            raise RuntimeError("找不到 PresetArch 定义")
        last = max(int(n) for n in re.findall(r"^    \w+ = (\d+)", enum.group(1), re.M))
        text = text[:enum.end()].rstrip("\n") + f"\n    moss = {last + 1}  {MARK}\n" + text[enum.end():]
    for name, value in PRESET_VALUES.items():
        block = re.search(rf"^{name} = \{{\n(.*?)^\}}", text, re.M | re.S)
        if block is not None and "PresetArch.moss" not in block.group(1):
            insert = f"    PresetArch.moss: {value},\n"
            text = text[:block.end() - 1] + insert + text[block.end() - 1:]
    if "PROCESS_HOOKS" not in text:
        text = text.rstrip("\n") + "\n" + HOOKS
    if "CHECKPOINT_PROVIDERS" not in text:
        text = text.rstrip("\n") + "\n" + PROVIDERS
    return text


def _patch_processing(text):
    if "PROCESS_HOOKS" in text:
        return text
    anchor = "    if p.scripts is not None:\n        p.scripts.before_process(p)\n"
    if anchor not in text:
        raise RuntimeError("找不到 process_images 的插入位置")
    hook = (
        anchor + f"\n    from modules_forge.presets import PROCESS_HOOKS  {MARK}\n\n"
        "    if (hook := PROCESS_HOOKS.get(opts.forge_preset)) is not None:\n        return hook(p)\n"
    )
    return text.replace(anchor, hook, 1)


def _patch_sd_models(text):
    if "CHECKPOINT_PROVIDERS" in text:
        return text
    anchor = "        checkpoint_info = CheckpointInfo(filename)\n        checkpoint_info.register()\n"
    if anchor not in text:
        raise RuntimeError("找不到 list_models 的插入位置")
    return text.replace(anchor, anchor + (
        f"\n    from modules_forge.presets import CHECKPOINT_PROVIDERS  {MARK}\n\n"
        "    for provider in CHECKPOINT_PROVIDERS:\n        provider()\n"
    ), 1)


def _patch_main_entry(text):
    if "preset_checkpoint_tiles" in text:
        return text
    imports = re.search(r"^from modules_forge\.presets import .*\n", text, re.M)
    old_list = "ckpt_list = shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)"
    old_value = 'gr.update(value=getattr(shared.opts, f"forge_checkpoint_{preset}", shared.opts.sd_model_checkpoint)),'
    if imports is None or old_list not in text:
        raise RuntimeError("找不到 Checkpoint 列表的位置")
    text = text[:imports.end()] + f"from modules_forge.presets import preset_checkpoint_tiles  {MARK}\n" + text[imports.end():]
    text = text.replace(old_list, "ckpt_list = preset_checkpoint_tiles(shared.opts.forge_preset, shared.opts.sd_checkpoint_dropdown_use_short)", 1)
    if old_value in text:  # 切换预设时同时刷新下拉框选项（找不到也不影响生成）
        indent = re.search(r"^( *)" + re.escape(old_value), text, re.M).group(1)
        text = text.replace(old_value, (
            f"gr.update(\n{indent}    value=getattr(shared.opts, f\"forge_checkpoint_{{preset}}\", shared.opts.sd_model_checkpoint),\n"
            f"{indent}    choices=sorted(preset_checkpoint_tiles(preset, shared.opts.sd_checkpoint_dropdown_use_short)),\n{indent}),"
        ), 1)
    return text


PATCHES = {
    "modules_forge/presets.py": _patch_presets,
    "modules/processing.py": _patch_processing,
    "modules/sd_models.py": _patch_sd_models,
    "modules_forge/main_entry.py": _patch_main_entry,
}


def apply(root):
    """返回 (已修改的文件列表, 说明)。不是 Forge Neo（没有 modules_forge/presets.py）时不做任何事。"""
    root = Path(root)
    if not (root / "modules_forge" / "presets.py").is_file():
        return [], "不是 Forge Neo（MOSS-Image 需要 Forge Neo），未做修改。"
    planned = {}
    for rel, patch in PATCHES.items():
        path = root / rel
        raw = path.read_bytes().decode("utf-8")
        newline = "\r\n" if "\r\n" in raw else "\n"
        text = raw.replace("\r\n", "\n")
        patched = patch(text)  # 任何一处找不到都在写入前报错，不会只改一半
        if patched != text:
            planned[path] = patched.replace("\n", newline)
    for path, content in planned.items():
        path.write_bytes(content.encode("utf-8"))
    return [str(p.relative_to(root)) for p in planned], "已是最新" if not planned else "已适配"


def is_applied():
    try:
        from modules_forge.presets import CHECKPOINT_PROVIDERS, PRESET_CHECKPOINTS, PROCESS_HOOKS, PresetArch  # noqa: F401

        return hasattr(PresetArch, "moss")
    except ImportError:
        return False
