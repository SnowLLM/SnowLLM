import pathlib
import re
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGETS = {"gfx1103": "110003", "gfx1150": "110500", "gfx1151": "110501",
           "gfx1152": "110502", "gfx1153": "110503"}
INSTALL = (ROOT / "install.sh").read_text()
assert INSTALL.count("\ncheck_machine\n") == 1
SOURCE = INSTALL.replace(
    "\ncheck_machine\n",
    '\nprintf "%s\\n" "$TARGET" "$GFX_TARGET" "$TORCH"\nreturn 0\n',
    1,
)


class InstallTargetsTest(unittest.TestCase):
    def run_selection(self, *args):
        return subprocess.run(["sh", "-s", "--", *args], input=SOURCE, text=True,
                              capture_output=True, timeout=5)

    def test_every_website_target_selects_matching_driver_id_and_torch_extra(self):
        page = (ROOT / "web/index.html").read_text()
        options = re.findall(r'<option value="([^"]+)" data-target="(gfx\d+)"', page)
        self.assertEqual({target for _, target in options}, set(TARGETS))
        self.assertEqual(len({model for model, _ in options}), len(options))
        for target, driver_id in TARGETS.items():
            with self.subTest(target=target):
                result = self.run_selection("--target", target)
                self.assertEqual(result.returncode, 0, result.stderr)
                arch, gfx, torch = result.stdout.splitlines()
                self.assertEqual((arch, gfx), (target, driver_id))
                self.assertIn(f"torch[device-{target}]==", torch)

    def test_default_is_gfx1151(self):
        result = self.run_selection()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[:2], ["gfx1151", "110501"])

    def test_last_target_wins_consistently(self):
        result = self.run_selection("--target", "gfx1103", "--target", "gfx1153")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[:2], ["gfx1153", "110503"])
        self.assertIn("device-gfx1153", result.stdout)

    def test_invalid_or_missing_target_fails_before_machine_checks(self):
        for args in [("--target",), ("--target", "gfx9999"), ("--unknown",)]:
            with self.subTest(args=args):
                result = self.run_selection(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
