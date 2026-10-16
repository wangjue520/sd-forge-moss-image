"""Layout JSON 校验：合法 JSON 原样作为 prompt 传入，只检查 bbox 与当前宽高。

官方格式（README / USERGUIDE）：
- ``bbox_coord_space: "abs_centralized"``：以画布中心为原点的像素坐标，x 向右、y 向下；
- ``bbox_format: "yxyx"``：``[y_min, x_min, y_max, x_max]``。
"""
from __future__ import annotations

import json
import math


def looks_like_layout(prompt):
    # 以 { 开头且含 "items" 才视为 Layout，避免误判 {a|b} 之类的通配符语法
    text = prompt.strip()
    return text.startswith("{") and '"items"' in text


def validate_layout(prompt, width, height):
    """prompt 不是 JSON 对象时返回 None；是 Layout 时校验通过返回条目数，否则抛 ValueError。"""
    if not looks_like_layout(prompt):
        return None
    try:
        layout = json.loads(prompt)
    except json.JSONDecodeError as error:
        raise ValueError(f"Layout JSON 解析失败（第 {error.lineno} 行第 {error.colno} 列）：{error.msg}") from None
    if not isinstance(layout, dict) or "items" not in layout:
        return None

    space = layout.get("bbox_coord_space")
    fmt = layout.get("bbox_format")
    if space == "abs" or fmt == "xyxy":
        raise ValueError(
            "这是 Layout 草稿格式（abs / xyxy）。MOSS-Image 需要 abs_centralized + yxyx，"
            "请先用官方 prompt-enhancer 的 compile_prompt.py 转换。"
        )
    if space != "abs_centralized":
        raise ValueError(f"bbox_coord_space 必须为 \"abs_centralized\"，当前为 {space!r}。")
    if fmt != "yxyx":
        raise ValueError(f"bbox_format 必须为 \"yxyx\"，当前为 {fmt!r}。")

    items = layout["items"]
    if not isinstance(items, list) or not items:
        raise ValueError("Layout 的 items 必须是非空数组。")
    half_w, half_h = width / 2, height / 2
    for index, item in enumerate(items):
        where = f"items[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where} 必须是对象。")
        box = item.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or not all(
            type(v) in (int, float) and math.isfinite(v) for v in box
        ):
            raise ValueError(f"{where}.bbox 必须是 4 个有限数字 [y_min, x_min, y_max, x_max]。")
        y1, x1, y2, x2 = box
        if not (y1 < y2 and x1 < x2):
            raise ValueError(f"{where}.bbox {box} 的宽高必须为正（y_min<y_max, x_min<x_max）。")
        if not (-half_h <= y1 and y2 <= half_h and -half_w <= x1 and x2 <= half_w):
            raise ValueError(
                f"{where}.bbox {box} 超出当前 {width}x{height} 画布"
                f"（y 范围 ±{half_h:g}，x 范围 ±{half_w:g}）。请让宽高与 Layout 设计尺寸一致。"
            )
        if item.get("label") == "text" and not str(item.get("text", "")).strip():
            raise ValueError(f"{where} 是 text 元素，必须提供非空的 text 字段。")
    return len(items)
