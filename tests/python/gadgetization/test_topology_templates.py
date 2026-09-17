import unittest

from security_tests.gadgetization import export_templates


class TopologyTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates = tuple(
            export_templates.build_template("nonlinear291", operation)
            for operation in ("r57", "nab", "and", "copy")
        )

    def test_checked_in_templates_exactly_match_in_memory_regeneration(self):
        expected = {
            template.filename: template.to_mpmct1() for template in self.templates
        }
        actual = {
            path.name: path.read_text(encoding="ascii")
            for path in export_templates.RUNTIME_TEMPLATE_DIR.glob("*.mpmct1")
        }
        self.assertEqual(actual, expected)

    def test_template_contracts(self):
        for template in self.templates:
            key = (template.variant, template.operation)
            with self.subTest(variant=template.variant, operation=template.operation):
                self.assertEqual(
                    len(template.gates), export_templates.EXPECTED_GATE_COUNTS[key]
                )
                self.assertLessEqual(
                    template.max_fanin,
                    export_templates.MAX_PHYSICAL_FANIN[template.variant],
                )
                if template.operation == "copy":
                    self.assertEqual(
                        template.layout["b_blocks"],
                        (tuple(range(15, 20)), tuple(range(20, 25))),
                    )


if __name__ == "__main__":
    unittest.main()
