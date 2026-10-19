"""MOSS-Image Layout JSON 工具：草稿编译、最终格式校验、提示词内嵌布局提取。

官方最终格式（USERGUIDE / ComfyUI ``MOSSImageLayoutPrompt``）：

- ``bbox_coord_space: "abs_centralized"``：以画布中心为原点的像素坐标，x 向右、y 向下；
- ``bbox_format: "yxyx"``：``[y_min, x_min, y_max, x_max]``。

可视化编辑器 / 官方 prompt-enhancer 的草稿格式：

- ``bbox_coord_space: "abs"``：左上角原点的像素坐标；
- ``bbox_format: "xyxy"``：``[x_min, y_min, x_max, y_max]``；
- 额外带 ``width`` / ``height``。

``compile_draft`` 把草稿转成最终格式；``normalize_layout_prompt`` 让直接粘贴草稿
到正向提示词也能出图，并保留 JSON 后面的 ``<lora:...>`` 等标签。
"""
from __future__ import annotations

import json
import math
from typing import Any

FINAL_SPACE = "abs_centralized"
FINAL_FORMAT = "yxyx"
DRAFT_SPACE = "abs"
DRAFT_FORMAT = "xyxy"
LABELS = ("object", "character", "text")


# --------------------------------------------------------------------------- JSON 提取


def split_layout(prompt: str) -> tuple[str | None, str]:
    """把 ``{...}`` 布局和后面的标签分开；不是布局时返回 ``(None, prompt)``。

    只接受出现在提示词开头的 JSON 对象。扫描时跳过字符串内的括号，因此
    caption 里出现 ``{`` / ``}`` 也不会截断。后面的 ``<lora:...>`` 原样保留。
    """
    text = str(prompt or "").strip()
    if not text.startswith("{"):
        return None, text
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[: index + 1], text[index + 1 :].strip()
    return None, text


def looks_like_layout(prompt: str) -> bool:
    layout, _ = split_layout(prompt)
    return layout is not None and '"items"' in layout


def extract_json(text: str) -> str | None:
    """从文本里找第一个完整的 JSON 对象（支持 Markdown 代码块 / 前后说明）。"""
    source = str(text or "")
    start = source.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(source)):
        char = source[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    return None


def _load_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Layout JSON 解析失败（第 {error.lineno} 行第 {error.colno} 列）：{error.msg}"
        ) from None
    if not isinstance(value, dict):
        raise ValueError("Layout JSON 必须是一个对象。")
    return value


# --------------------------------------------------------------------------- 数值与 bbox


def _number(value: Any, where: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{where} 必须是有限数字。")
    return float(value)


def _positive_dimension(value: Any, where: str) -> int:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{where} 必须是正数。")
    return int(round(value))


def _item_box(item: dict[str, Any], where: str) -> list[Any]:
    box = item.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"{where}.bbox 必须是 4 个数字。")
    return box


def _final_box(
    box: list[Any], width: int, height: int, where: str
) -> list[int]:
    """草稿左上角 ``[x1,y1,x2,y2]`` -> 最终居中 ``[y1,x1,y2,x2]``。"""
    x1, y1, x2, y2 = (_number(value, f"{where}.bbox") for value in box)
    if not (x1 < x2 and y1 < y2):
        raise ValueError(f"{where}.bbox 必须满足 x1<x2 且 y1<y2。")
    if not (0 <= x1 and x2 <= width and 0 <= y1 and y2 <= height):
        raise ValueError(
            f"{where}.bbox {box} 超出 {width}x{height} 画布（左上角原点）。"
        )
    half_w, half_h = width / 2, height / 2
    return [
        math.floor(y1 - half_h),
        math.floor(x1 - half_w),
        math.ceil(y2 - half_h),
        math.ceil(x2 - half_w),
    ]


def _normalize_final_box(
    box: list[Any], width: int | None, height: int | None, where: str
) -> list[int]:
    y1, x1, y2, x2 = (_number(value, f"{where}.bbox") for value in box)
    if not (y1 < y2 and x1 < x2):
        raise ValueError(f"{where}.bbox 必须满足 y1<y2 且 x1<x2（yxyx）。")
    values = [math.floor(y1), math.floor(x1), math.ceil(y2), math.ceil(x2)]
    if width and height:
        half_w, half_h = width / 2, height / 2
        if not (-half_h <= values[0] and values[2] <= half_h and -half_w <= values[1] and values[3] <= half_w):
            raise ValueError(
                f"{where}.bbox {box} 超出当前 {width}x{height} 画布"
                f"（y 范围 ±{half_h:g}，x 范围 ±{half_w:g}）。"
            )
    return values


# --------------------------------------------------------------------------- 编译


def _copy_summary(summary: Any) -> dict[str, Any]:
    if summary is None:
        return {"caption": ""}
    if not isinstance(summary, dict):
        raise ValueError("summary 必须是对象。")
    result: dict[str, Any] = {"caption": str(summary.get("caption", "") or "")}
    meta = summary.get("meta")
    if meta is not None:
        if not isinstance(meta, list) or not all(isinstance(item, str) for item in meta):
            raise ValueError("summary.meta 必须是字符串数组。")
        result["meta"] = list(meta)
    canvas = summary.get("canvas")
    if canvas is not None:
        if canvas not in ("single", "multi_panel", "collage"):
            raise ValueError("summary.canvas 只能是 single / multi_panel / collage。")
        result["canvas"] = canvas
    palette = summary.get("palette")
    if palette is not None:
        if not isinstance(palette, list) or not all(isinstance(item, str) for item in palette):
            raise ValueError("summary.palette 必须是字符串数组。")
        result["palette"] = list(palette)
    return result


def _copy_item(
    item: Any,
    index: int,
    *,
    width: int | None,
    height: int | None,
    source: str,
    characters: set[int],
) -> dict[str, Any]:
    where = f"items[{index}]"
    if not isinstance(item, dict):
        raise ValueError(f"{where} 必须是对象。")

    label = item.get("label")
    if label is None:
        # Create Bounding Boxes 的草稿用 type: obj / text；character 由单独的编号列表标记
        raw_type = str(item.get("type", "obj")).lower()
        label = "text" if raw_type == "text" else "object"
    label = str(label).lower()
    if label == "obj":
        label = "object"
    if label not in LABELS:
        raise ValueError(f"{where}.label 只能是 object / character / text。")
    if index + 1 in characters and label == "object":
        label = "character"

    caption = item.get("caption", item.get("desc", ""))
    if caption is None:
        caption = ""
    if not isinstance(caption, str):
        raise ValueError(f"{where}.caption 必须是字符串。")
    result: dict[str, Any] = {"label": label, "caption": caption}

    if label == "text":
        text = item.get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{where} 是 text 元素，必须提供非空的 text 字段。")
        result["text"] = text
    elif item.get("text"):
        # 非 text 元素的 text 字段没有意义，直接丢弃，避免模型误判
        pass

    tags = item.get("tags")
    if tags is not None:
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"{where}.tags 必须是字符串数组。")
        result["tags"] = list(tags)
    palette = item.get("palette")
    if palette is not None:
        if not isinstance(palette, list) or not all(isinstance(color, str) for color in palette):
            raise ValueError(f"{where}.palette 必须是字符串数组。")
        result["palette"] = list(palette)

    box = _item_box(item, where)
    if source == "draft":
        if width is None or height is None:
            raise ValueError("草稿缺少 width / height，无法把左上角坐标转换成中心坐标。")
        result["bbox"] = _final_box(box, width, height, where)
    else:
        result["bbox"] = _normalize_final_box(box, width, height, where)
    return result


def _characters(layout: dict[str, Any]) -> set[int]:
    raw = layout.get("character_boxes", "")
    if raw in (None, ""):
        return set()
    if isinstance(raw, str):
        import re

        parts = re.findall(r"\d+", raw)
    elif isinstance(raw, (list, tuple)):
        parts = [str(value) for value in raw]
    else:
        raise ValueError("character_boxes 必须是字符串或数组。")
    try:
        values = {int(value) for value in parts}
    except (TypeError, ValueError):
        raise ValueError("character_boxes 里只能有框编号。") from None
    if any(value < 1 for value in values):
        raise ValueError("character_boxes 的编号从 1 开始。")
    return values


def compile_draft(
    layout: dict[str, Any], width: int | None = None, height: int | None = None
) -> dict[str, Any]:
    """把草稿或最终 Layout 统一成最终格式（幂等）。

    ``width`` / ``height`` 是当前生成画布；草稿自己带尺寸时以草稿为准，
    但会校验它与当前画布一致。最终格式没有尺寸字段，用传入值做越界检查。
    """
    if not isinstance(layout, dict):
        raise ValueError("Layout 必须是一个对象。")
    space = layout.get("bbox_coord_space")
    fmt = layout.get("bbox_format")
    source: str
    if space == DRAFT_SPACE and fmt == DRAFT_FORMAT:
        source = "draft"
        draft_w = layout.get("width")
        draft_h = layout.get("height")
        if draft_w is None or draft_h is None:
            if width is None or height is None:
                raise ValueError("草稿缺少 width / height，无法确定画布尺寸。")
            width, height = int(width), int(height)
        else:
            draft_width = _positive_dimension(draft_w, "width")
            draft_height = _positive_dimension(draft_h, "height")
            if width is not None and int(width) != draft_width:
                raise ValueError(
                    f"草稿画布宽度是 {draft_width}，当前生成宽度是 {int(width)}；"
                    "请把宽高改成一致，或在编辑器里按当前尺寸重新编译。"
                )
            if height is not None and int(height) != draft_height:
                raise ValueError(
                    f"草稿画布高度是 {draft_height}，当前生成高度是 {int(height)}；"
                    "请把宽高改成一致，或在编辑器里按当前尺寸重新编译。"
                )
            width, height = draft_width, draft_height
            if width % 16 or height % 16:
                raise ValueError("Layout 的 width / height 必须是 16 的倍数。")
    elif space == FINAL_SPACE and fmt == FINAL_FORMAT:
        source = "final"
        if width is not None:
            width = int(width)
        if height is not None:
            height = int(height)
    else:
        raise ValueError(
            "Layout 需要 bbox_coord_space=abs + bbox_format=xyxy（草稿），"
            "或 abs_centralized + yxyx（最终格式）。"
        )

    items = layout.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Layout 的 items 必须是非空数组。")
    characters = _characters(layout)
    compiled_items = [
        _copy_item(
            item,
            index,
            width=width,
            height=height,
            source=source,
            characters=characters,
        )
        for index, item in enumerate(items)
    ]
    return {
        "summary": _copy_summary(layout.get("summary")),
        "items": compiled_items,
        "bbox_coord_space": FINAL_SPACE,
        "bbox_format": FINAL_FORMAT,
    }


def compile_json(
    layout: dict[str, Any], width: int | None = None, height: int | None = None
) -> str:
    """编译并序列化成可直接放进正向提示词的紧凑 JSON。"""
    return json.dumps(
        compile_draft(layout, width, height),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def compile_text(
    text: str, width: int | None = None, height: int | None = None
) -> str:
    """编译一段 JSON 文本；不是布局或解析失败时抛出 ``ValueError``。"""
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("Layout JSON 为空。")
    candidate = extract_json(raw)
    if candidate is None:
        raise ValueError("没有找到完整的 Layout JSON 对象。")
    return compile_json(_load_object(candidate), width, height)


# --------------------------------------------------------------------------- 校验 / 提示词处理


def validate_layout(prompt: str, width: int, height: int) -> int | None:
    """prompt 不是布局时返回 ``None``；是最终格式时校验通过返回条目数。

    草稿会抛出 ``ValueError``，并提示使用面板的「编译草稿」或直接粘贴
    （生成时 ``normalize_layout_prompt`` 会自动编译）。
    """
    layout_text, _ = split_layout(prompt)
    if layout_text is None or '"items"' not in layout_text:
        return None
    layout = _load_object(layout_text)
    space = layout.get("bbox_coord_space")
    fmt = layout.get("bbox_format")
    if space == DRAFT_SPACE or fmt == DRAFT_FORMAT:
        raise ValueError(
            "这是 Layout 草稿格式（abs / xyxy）。生成时会自动编译；"
            "也可以先在 MOSS 面板的 Layout 编辑器里点「编译草稿」。"
        )
    if space != FINAL_SPACE:
        raise ValueError(f'bbox_coord_space 必须为 "{FINAL_SPACE}"，当前为 {space!r}。')
    if fmt != FINAL_FORMAT:
        raise ValueError(f'bbox_format 必须为 "{FINAL_FORMAT}"，当前为 {fmt!r}。')
    items = layout.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Layout 的 items 必须是非空数组。")
    characters = _characters(layout)
    for index, item in enumerate(items):
        _copy_item(
            item,
            index,
            width=width,
            height=height,
            source="final",
            characters=characters,
        )
    return len(items)


def normalize_layout_prompt(prompt: str, width: int, height: int) -> tuple[str, str | None]:
    """把提示词开头的草稿自动编译成最终格式，保留后面的 LoRA 等标签。

    返回 ``(新提示词, 提示信息)``；不是布局时原样返回。
    """
    layout_text, trailing = split_layout(prompt)
    if layout_text is None or '"items"' not in layout_text:
        return prompt, None
    layout = _load_object(layout_text)
    space = layout.get("bbox_coord_space")
    fmt = layout.get("bbox_format")
    if space == FINAL_SPACE and fmt == FINAL_FORMAT:
        validate_layout(prompt, width, height)
        return prompt, None
    compiled = compile_json(layout, width, height)
    message = "草稿 Layout 已自动编译为 abs_centralized / yxyx。"
    return (compiled + ((" " + trailing) if trailing else "")), message


def rescale_layout(prompt: str, width: int, height: int, target_w: int, target_h: int) -> str:
    """Hires 第二遍按比例缩放最终 Layout 的 bbox；不是布局时原样返回。"""
    layout_text, trailing = split_layout(prompt)
    if layout_text is None or '"items"' not in layout_text:
        return prompt
    layout = _load_object(layout_text)
    if layout.get("bbox_coord_space") != FINAL_SPACE or layout.get("bbox_format") != FINAL_FORMAT:
        return prompt
    sx, sy = target_w / float(width), target_h / float(height)
    for item in layout.get("items", []):
        if not isinstance(item, dict) or not isinstance(item.get("bbox"), list):
            continue
        y1, x1, y2, x2 = item["bbox"]
        item["bbox"] = [
            math.floor(y1 * sy),
            math.floor(x1 * sx),
            math.ceil(y2 * sy),
            math.ceil(x2 * sx),
        ]
    result = json.dumps(layout, ensure_ascii=False, separators=(",", ":"))
    return result + ((" " + trailing) if trailing else "")
