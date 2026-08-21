from __future__ import annotations

import unittest

from testcase.build_eval_set import (
    balanced_sample,
    build_manifest,
    changed_files,
    changed_lines,
    infer_category,
    infer_difficulty,
    parse_json_list,
    summarize,
)


class TestPatchClassification(unittest.TestCase):
    def test_counts_changed_files_and_lines(self):
        patch = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@
-if value == 1:
+if value != 1:
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@
-return 0
+return 1
"""
        self.assertEqual(changed_files(patch), 2)
        self.assertEqual(changed_lines(patch), 4)

    def test_infers_comparison_category(self):
        patch = "-if value == 1:\n+if value != 1:\n"
        self.assertEqual(infer_category("wrong result", patch), "comparison_logic")

    def test_infers_keyword_category_before_generic_patch_category(self):
        self.assertEqual(infer_category("Fix an import error"), "dependency_import")

    def test_difficulty_bands(self):
        self.assertEqual(infer_difficulty("-x = 1\n+x = 2\n"), "easy")
        medium_patch = "diff --git a/a.py b/a.py\n" + "".join(
            f"-old{i}\n+new{i}\n" for i in range(11)
        )
        self.assertEqual(infer_difficulty(medium_patch), "medium")


class TestSelection(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "source": "sample",
                "instance_id": f"task-{index}",
                "repository": f"repo-{index % 3}",
                "category": f"category-{index % 4}",
                "difficulty": ("easy", "medium", "hard")[index % 3],
            }
            for index in range(20)
        ]

    def test_balanced_sample_is_deterministic(self):
        first = balanced_sample(self.rows, 10, 42, lambda row: row["repository"])
        second = balanced_sample(self.rows, 10, 42, lambda row: row["repository"])
        self.assertEqual(
            [row["instance_id"] for row in first],
            [row["instance_id"] for row in second],
        )

    def test_balanced_sample_rejects_oversized_request(self):
        with self.assertRaises(RuntimeError):
            balanced_sample(self.rows, 21, 42, lambda row: row["repository"])

    def test_summary_counts_dimensions(self):
        summary = summarize(self.rows[:3])
        self.assertEqual(summary["by_source"], {"sample": 3})
        self.assertEqual(sum(summary["by_category"].values()), 3)

    def test_parse_json_list_accepts_dataset_encodings(self):
        self.assertEqual(parse_json_list('["a", "b"]'), ["a", "b"])
        self.assertEqual(parse_json_list(["a"]), ["a"])
        self.assertEqual(parse_json_list(""), [])

    def test_build_manifest_rejects_empty_suite_without_downloading(self):
        args = type(
            "Args",
            (),
            {
                "cache_dir": None,
                "quixbugs_count": 0,
                "bugsinpy_count": 0,
                "swebench_count": 0,
            },
        )()
        with self.assertRaises(ValueError):
            build_manifest(args)


if __name__ == "__main__":
    unittest.main()
