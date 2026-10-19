from __future__ import annotations

import json
import unittest

from lib_moss_image import layout


def draft(**overrides):
    value = {
        "width": 1024,
        "height": 1024,
        "bbox_coord_space": "abs",
        "bbox_format": "xyxy",
        "summary": {"caption": "A poster", "meta": ["poster"], "canvas": "single"},
        "items": [
            {"label": "object", "caption": "a red fox", "bbox": [100, 200, 300, 400]},
            {"label": "text", "caption": "title", "text": "FOX", "bbox": [50, 50, 500, 120]},
        ],
    }
    value.update(overrides)
    return value


class LayoutCompileTest(unittest.TestCase):
    def test_draft_to_final(self):
        result = layout.compile_draft(draft(), 1024, 1024)
        self.assertEqual(result["bbox_coord_space"], "abs_centralized")
        self.assertEqual(result["bbox_format"], "yxyx")
        self.assertEqual(result["items"][0]["bbox"], [-312, -412, -112, -212])
        self.assertEqual(result["items"][1]["label"], "text")
        self.assertEqual(result["items"][1]["text"], "FOX")

    def test_idempotent(self):
        once = layout.compile_draft(draft(), 1024, 1024)
        twice = layout.compile_draft(once, 1024, 1024)
        self.assertEqual(once, twice)

    def test_official_draft_type_desc(self):
        value = draft()
        value["items"] = [
            {"type": "obj", "desc": "a red fox", "bbox": [100, 200, 300, 400]},
            {"type": "text", "desc": "title", "text": "FOX", "bbox": [50, 50, 500, 120]},
        ]
        result = layout.compile_draft(value, 1024, 1024)
        self.assertEqual(result["items"][0]["label"], "object")
        self.assertEqual(result["items"][0]["caption"], "a red fox")
        self.assertEqual(result["items"][1]["label"], "text")

    def test_character_boxes(self):
        value = draft(character_boxes="1")
        result = layout.compile_draft(value, 1024, 1024)
        self.assertEqual(result["items"][0]["label"], "character")

    def test_text_requires_text(self):
        value = draft()
        value["items"][1]["text"] = " "
        with self.assertRaisesRegex(ValueError, "非空"):
            layout.compile_draft(value, 1024, 1024)

    def test_size_mismatch(self):
        with self.assertRaisesRegex(ValueError, "宽度"):
            layout.compile_draft(draft(), 768, 1024)

    def test_bbox_out_of_canvas(self):
        value = draft()
        value["items"][0]["bbox"] = [900, 900, 1100, 1000]
        with self.assertRaisesRegex(ValueError, "超出"):
            layout.compile_draft(value, 1024, 1024)

    def test_mixed_coordinate_format_rejected(self):
        value = draft()
        value["bbox_format"] = "yxyx"
        with self.assertRaisesRegex(ValueError, "bbox_coord_space"):
            layout.compile_draft(value, 1024, 1024)

    def test_final_bounds(self):
        value = draft()
        final = layout.compile_draft(value, 1024, 1024)
        self.assertEqual(layout.validate_layout(json.dumps(final), 1024, 1024), 2)
        final["items"][0]["bbox"] = [-600, 0, 10, 10]
        with self.assertRaisesRegex(ValueError, "超出"):
            layout.validate_layout(json.dumps(final), 1024, 1024)

    def test_split_braces_in_caption(self):
        value = draft()
        value["summary"]["caption"] = 'A sign reading "{FOX}"'
        text = json.dumps(value) + " <lora:test:1>"
        layout_text, trailing = layout.split_layout(text)
        self.assertTrue(layout_text.startswith("{"))
        self.assertEqual(trailing, "<lora:test:1>")
        compiled, message = layout.normalize_layout_prompt(text, 1024, 1024)
        self.assertTrue(compiled.endswith("<lora:test:1>"))
        self.assertIn("自动编译", message)

    def test_normalize_final_keeps_text(self):
        final = layout.compile_json(draft(), 1024, 1024)
        normalized, message = layout.normalize_layout_prompt(final, 1024, 1024)
        self.assertEqual(normalized, final)
        self.assertIsNone(message)

    def test_rescale(self):
        final = layout.compile_json(draft(), 1024, 1024)
        scaled = layout.rescale_layout(final, 1024, 1024, 2048, 2048)
        value = json.loads(scaled)
        self.assertEqual(value["items"][0]["bbox"], [-624, -824, -224, -424])

    def test_not_layout(self):
        self.assertIsNone(layout.validate_layout("plain prompt", 1024, 1024))
        self.assertEqual(layout.normalize_layout_prompt("plain prompt", 1024, 1024), ("plain prompt", None))

    def test_compile_text_from_markdown(self):
        raw = "```json\n" + json.dumps(draft()) + "\n```"
        compiled = layout.compile_text(raw, 1024, 1024)
        value = json.loads(compiled)
        self.assertEqual(value["bbox_coord_space"], "abs_centralized")


if __name__ == "__main__":
    unittest.main()
