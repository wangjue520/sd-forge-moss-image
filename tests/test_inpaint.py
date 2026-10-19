from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from lib_moss_image import inpaint


class InpaintCropTest(unittest.TestCase):
    def test_mask_bbox(self):
        mask = np.zeros((100, 120), dtype=np.float32)
        mask[20:50, 30:70] = 1
        self.assertEqual(inpaint.mask_bbox(mask), (30, 20, 70, 50))

    def test_empty_mask(self):
        self.assertIsNone(inpaint.mask_bbox(np.zeros((10, 10), dtype=np.float32)))

    def test_expand_and_clamp(self):
        self.assertEqual(inpaint.expand_bbox((10, 20, 30, 40), 8, 100, 100), (2, 12, 38, 48))
        self.assertEqual(inpaint.expand_bbox((1, 2, 98, 99), 20, 100, 100), (0, 0, 100, 100))

    def test_crop_and_paste(self):
        image = Image.new("RGB", (200, 200), (10, 20, 30))
        mask = Image.new("L", (200, 200), 0)
        mask.paste(255, (50, 60, 150, 160))
        bbox = inpaint.expand_bbox(inpaint.mask_bbox(np.asarray(mask), 0.5), 10, 200, 200)
        crop, crop_mask = inpaint.crop_pair(image, mask, bbox, 128, 128)
        self.assertEqual(crop.size, (128, 128))
        self.assertEqual(crop_mask.size, (128, 128))

        generated = Image.new("RGB", (128, 128), (200, 100, 50))
        result = inpaint.paste_crop(image, generated, bbox)
        self.assertEqual(result.size, image.size)
        self.assertEqual(result.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(result.getpixel((100, 100)), (200, 100, 50))

    def test_composite(self):
        original = Image.new("RGB", (10, 10), (0, 0, 0))
        generated = Image.new("RGB", (10, 10), (255, 255, 255))
        mask = Image.new("L", (10, 10), 0)
        mask.paste(255, (5, 0, 10, 10))
        result = inpaint.composite_with_mask(original, generated, mask)
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(result.getpixel((9, 9)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
