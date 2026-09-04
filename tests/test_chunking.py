import unittest

from src.build_corpus import (
    PASSAGE_MAX_TOKENS,
    Example,
    HowTo,
    _app_access_patterns,
    _best_description,
    _description_matches_subject,
    _python_node_chunks,
    _rough_token_count,
    build_passages,
)


class PassageChunkingTests(unittest.TestCase):
    def test_markdown_is_heading_aware_and_bounded(self):
        code = "\n".join(f"value_{i} = gom.app.project.parts[{i}]" for i in range(600))
        howto = HowTo(
            slug="demo.guide",
            title="Demo guide",
            source_file="doc/howtos/demo/guide.md",
            content=(
                "# Demo guide\n\nIntro text.\n\n"
                "## Select parts\n\nUse the project parts collection.\n\n"
                f"```python\n{code}\n```\n\n"
                "## Export\n\nExport the result."
            ),
        )

        passages = build_passages({howto.slug: howto}, {})

        self.assertGreater(len(passages), 3)
        self.assertTrue(any("Select parts" in p.title for p in passages.values()))
        self.assertTrue(any("Export" in p.title for p in passages.values()))
        self.assertTrue(all(
            _rough_token_count(p.content) <= PASSAGE_MAX_TOKENS
            for p in passages.values()
        ))
        for passage in passages.values():
            if passage.content.startswith("```python"):
                self.assertTrue(passage.content.endswith("```"))

    def test_large_python_class_uses_method_boundaries_without_losing_tail(self):
        repeated = "\n".join("        self.value += 1" for _ in range(500))
        source = (
            "import gom\n\n"
            "class Worker:\n"
            "    def run(self):\n"
            "        self.value = 0\n"
            f"{repeated}\n"
            "    sentinel = 42\n"
        )

        chunks = _python_node_chunks(source)
        labels = [label for label, _ in chunks]
        combined = "\n".join(content for _, content in chunks)

        self.assertTrue(any("Worker.run" in label for label in labels))
        self.assertIn("sentinel = 42", combined)
        self.assertTrue(all(
            _rough_token_count(content) <= PASSAGE_MAX_TOKENS
            for _, content in chunks
        ))

    def test_example_scripts_become_searchable_passages(self):
        example = Example(
            name="PartReport",
            category="reports",
            path="AppExamples/reports/PartReport",
            documentation="# Part report\n\n## Usage\n\nCreate the report.",
            scripts={
                "scripts/main.py": (
                    "import gom\n\n"
                    "def create_report():\n"
                    "    return gom.app.project.parts['Part 1']\n"
                )
            },
        )

        passages = build_passages({}, {example.name: example})

        self.assertTrue(any(p.kind == "example_doc" for p in passages.values()))
        script = [p for p in passages.values() if p.kind == "example_script"]
        self.assertEqual(len(script), 1)
        self.assertIn("gom.app.project.parts", script[0].api_mentions)
        self.assertEqual(script[0].source_file,
                         "AppExamples/reports/PartReport/scripts/main.py")


class InferenceQualityTests(unittest.TestCase):
    def test_access_patterns_are_balanced_and_normalized(self):
        source = (
            "gom.app.project.parts[f'Point {i+1}.epsX']"
            ".in_stage[gom.app.project.stages[j].index]"
        )

        patterns = _app_access_patterns(source, "gom.app.project.parts")

        self.assertEqual(patterns, ["['name'].in_stage[expression]"])
        self.assertTrue(all(p.count("[") == p.count("]") for p in patterns))

    def test_unrelated_prose_is_not_attribute_evidence(self):
        bad = "This function computes a 3d point in a 2d image."
        good = "Access the actual elements stored in the current project."

        self.assertFalse(_description_matches_subject(
            "gom.app.project.actual_elements", bad
        ))
        self.assertTrue(_description_matches_subject(
            "gom.app.project.actual_elements", good
        ))
        self.assertEqual(
            _best_description(
                ["5. Add a cell for configuring the connection.",
                 "Read the current build date."],
                "gom.app.application_build_information.date",
            ),
            "Read the current build date.",
        )


if __name__ == "__main__":
    unittest.main()
