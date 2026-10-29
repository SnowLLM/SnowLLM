from html.parser import HTMLParser
import json
from pathlib import Path
import runpy
import unittest


ROOT = Path(__file__).resolve().parent.parent
BUILDER = runpy.run_path(str(ROOT / "scripts" / "build-benchmarks.py"))
MESSAGES = json.loads((ROOT / "web" / "locales" / "en.json").read_text())


class ResultsParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = {}
        self.elements = {}
        self.current = None
        self.cells = None
        self.cell = None
        self.inline = ""
        self.in_inline = False
        self.result_id = None
        self.result_rows = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.elements[attrs["id"]] = attrs
        if tag == "details":
            self.result_id = attrs.get("id")
        if tag == "tr" and "data-series-key" in attrs:
            self.current = (attrs["data-series-key"], int(attrs["data-isl-target"]))
            self.cells = []
        elif tag in ("th", "td") and self.current is not None:
            self.cell = ""
        elif tag == "script" and attrs.get("id") == "benchmarks-data":
            self.in_inline = True

    def handle_data(self, data):
        if self.in_inline:
            self.inline += data
        elif self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.cell is not None:
            self.cells.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.current is not None:
            if self.current in self.rows:
                raise AssertionError(f"duplicate measurement: {self.current}")
            self.rows[self.current] = self.cells
            self.result_rows[self.current] = self.result_id
            self.current = None
        elif tag == "details":
            self.result_id = None
        elif tag == "script":
            self.in_inline = False


class BenchmarkPageTests(unittest.TestCase):
    def test_hardware_groups_keep_their_own_measurements(self):
        series, skipped = BUILDER["build_series"](BUILDER["merged_rows"]())
        self.assertFalse(set(skipped) & BUILDER["GFX1103_LABELS"])
        parser = ResultsParser()
        content = BUILDER["render_benchmark_content"]({"series": series}, MESSAGES)
        parser.feed(content)
        secondary = [s for s in series if s["platform"] == "gfx1103"]
        self.assertEqual({s["key"] for s in secondary}, BUILDER["GFX1103_LABELS"])
        self.assertEqual({s["engine"] for s in secondary}, {"SnowLLM", "llama.cpp"})
        for s in series:
            expected = "benchmark-results-780m" if s["platform"] == "gfx1103" else "benchmark-results"
            for p in s["points"]:
                self.assertEqual(parser.result_rows[s["key"], p["isl_target"]], expected)
        self.assertLess(content.index('id="benchmark-results"'), content.index('id="benchmarks-780m"'))
        for s in secondary:
            self.assertEqual([p["isl_target"] for p in s["points"]], [1024, 8192, 32768, 122880])

    def test_cached_scripts_can_render_charts_without_removing_hidden_attributes(self):
        parser = ResultsParser()
        parser.feed((ROOT / "web" / "benchmark" / "index.html").read_text())
        for element in ["benchmarks-picker", "benchmarks-charts", "benchmarks-780m-picker", "benchmarks-780m-charts"]:
            self.assertNotIn("hidden", parser.elements[element])
        for element, platform in [("benchmarks-picker", "gfx1151"), ("benchmarks-780m-picker", "gfx1103")]:
            self.assertIn("benchmark-picker", parser.elements[element]["class"].split())
            self.assertEqual(parser.elements[element]["data-platform"], platform)

    def test_every_published_measurement_is_readable_without_javascript(self):
        series, _ = BUILDER["build_series"](BUILDER["merged_rows"]())
        data = {"schema": 2, "series": series, "rows": []}
        parser = ResultsParser()
        parser.feed(BUILDER["render_benchmark_content"](data, MESSAGES))
        expected = {(s["key"], p["isl_target"]) for s in series for p in s["points"]}
        self.assertEqual(set(parser.rows), expected)
        self.assertEqual(json.loads(parser.inline), data)
        for s in series:
            for p in s["points"]:
                cells = parser.rows[s["key"], p["isl_target"]]
                self.assertEqual(cells[:2], [s["engine"], s["arm"]])
                self.assertEqual(int(cells[2].replace(",", "")), p["isl_target"])
                self.assertEqual(int(cells[3].replace(",", "")), p["isl"])
                for cell, key in zip(cells[4:], ["decode", "prefill", "ttft", "step_ms", "al"]):
                    if p[key] is None:
                        self.assertEqual(cell, "—")
                    else:
                        self.assertEqual(float(cell), p[key])

    def test_html_and_inline_json_preserve_strings_without_creating_markup(self):
        data = {"series": [{
            "key": 'test"<&', "model": "<Model & name>",
            "engine": '</script><script>alert("test")</script>', "arm": "no spec & extra",
            "points": [{"isl_target": 1024, "isl": 1030, "decode": 12.5, "prefill": 50, "ttft": 1, "step_ms": None, "al": None}],
        }]}
        content = BUILDER["render_benchmark_content"](data, MESSAGES)
        self.assertIn("&lt;Model &amp; name&gt;", content)
        self.assertEqual(content.count("<script"), 1)
        parser = ResultsParser()
        parser.feed(content)
        self.assertEqual(json.loads(parser.inline), data)
        self.assertEqual(parser.rows['test"<&', 1024][:2], [data["series"][0]["engine"], "no spec & extra"])

    def test_rebuilding_replaces_only_the_generated_region(self):
        template = "before\n" + BUILDER["START"] + "stale data" + BUILDER["END"] + "\nafter"
        data = {"series": []}
        rendered = BUILDER["render_benchmark_page"](template, data, MESSAGES)
        self.assertTrue(rendered.startswith("before\n"))
        self.assertTrue(rendered.endswith("\nafter"))
        self.assertNotIn("stale data", rendered)
        self.assertEqual(BUILDER["render_benchmark_page"](rendered, data, MESSAGES), rendered)
        for invalid in ["missing region", template + BUILDER["START"], template + BUILDER["END"]]:
            with self.assertRaises(ValueError):
                BUILDER["render_benchmark_page"](invalid, data, MESSAGES)


if __name__ == "__main__":
    unittest.main()
