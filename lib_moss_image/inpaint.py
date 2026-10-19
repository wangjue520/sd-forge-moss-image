"""局部重绘的裁剪辅助函数（PIL / NumPy，无 Forge 依赖）。

Forge 的「仅重绘蒙版区域」不是把整张图重新生成，而是：

1. 求蒙版的包围盒，向外留一圈上下文 padding；
2. 裁出这一块，缩放到生成尺寸，只对这一块做 img2img；
3. 把结果缩回包围盒大小，贴回原图；
4. 最后用原始蒙版做一次合成，蒙版外像素保持不变。

这样小区域也能用完整的 1024x1024 分辨率重绘，速度也比整图快。
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image


def mask_bbox(mask: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int, int] | None:
    """返回蒙版的 ``(x1, y1, x2, y2)``（右/下为开区间）；空蒙版返回 None。"""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be a 2D array")
    ys, xs = np.where(array > float(threshold))
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_bbox(
    bbox: Iterable[int], padding: int, width: int, height: int
) -> tuple[int, int, int, int]:
    """按像素向外扩一圈，并夹在画布内。"""
    x1, y1, x2, y2 = (int(value) for value in bbox)
    padding = max(0, int(padding))
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(int(width), x2 + padding)
    y2 = min(int(height), y2 + padding)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("empty crop bbox")
    return x1, y1, x2, y2


def crop_pair(
    image: Image.Image,
    mask: Image.Image,
    bbox: Iterable[int],
    target_w: int,
    target_h: int,
) -> tuple[Image.Image, Image.Image]:
    """裁出图像/蒙版并缩放到生成尺寸。"""
    x1, y1, x2, y2 = (int(value) for value in bbox)
    crop = image.crop((x1, y1, x2, y2)).convert("RGB")
    crop_mask = mask.crop((x1, y1, x2, y2)).convert("L")
    crop = crop.resize((int(target_w), int(target_h)), Image.LANCZOS)
    crop_mask = crop_mask.resize((int(target_w), int(target_h)), Image.BILINEAR)
    return crop, crop_mask


def paste_crop(
    base: Image.Image, generated: Image.Image, bbox: Iterable[int]
) -> Image.Image:
    """把生成的小图缩回包围盒大小，贴到原图副本上。"""
    x1, y1, x2, y2 = (int(value) for value in bbox)
    patch = generated.convert("RGB").resize((max(1, x2 - x1), max(1, y2 - y1)), Image.LANCZOS)
    result = base.copy()
    result.paste(patch, (x1, y1))
    return result


def composite_with_mask(
    original: Image.Image, generated: Image.Image, mask: Image.Image
) -> Image.Image:
    """用原始蒙版做最终合成；mask 为 L 模式，255 = 生成，0 = 保留。"""
    if original.size != generated.size or original.size != mask.size:
        raise ValueError("original / generated / mask must have the same size")
    return Image.composite(generated.convert("RGB"), original.convert("RGB"), mask.convert("L"))
