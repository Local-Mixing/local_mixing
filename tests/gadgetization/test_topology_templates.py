import unittest

from gadgetization import export_templates


class TopologyTemplateTests(unittest.TestCase):
    def test_checked_in_templates_exactly_match_in_memory_regeneration(self):
        expected = export_templates.render_all_templates()
        actual = {
            path.name: path.read_text(encoding="ascii")
            for path in export_templates.TEMPLATE_DIR.glob("*.mpmct1")
        }
        self.assertEqual(actual, expected)

    def test_template_contracts(self):
        templates = export_templates.build_all_templates()
        self.assertEqual(len(templates), 8)
        for template in templates:
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
