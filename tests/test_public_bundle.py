import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('public_bundle', ROOT / 'scripts/check-public-bundle.py')
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class PublicBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git('init', '-q', '-b', 'main')
        (self.root / 'README').write_text('public fixture\n')
        self.git('add', 'README')
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                 'commit', '-qm', 'fixture')
        self.git('tag', 'v1.0.0')
        self.git('tag', 'v1.0.1.dev1')
        self.head = self.git('rev-parse', 'HEAD').strip()

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], text=True)

    def bundle(self, *refs):
        path = self.root / 'release.bundle'
        self.git('bundle', 'create', str(path), *refs)
        return path

    def test_accepts_main_and_version_tags(self):
        path = self.bundle('--all')
        refs = CHECKER.check(path, self.head)
        self.assertEqual(set(refs), {'HEAD', 'refs/heads/main', 'refs/tags/v1.0.0',
                                    'refs/tags/v1.0.1.dev1'})

    def test_rejects_backup_tag_even_with_public_content(self):
        self.git('tag', 'backup-worktree-fixture')
        with self.assertRaisesRegex(ValueError, 'non-release ref'):
            CHECKER.check(self.bundle('--all'))

    def test_rejects_private_branch(self):
        self.git('branch', 'private-experiment')
        with self.assertRaisesRegex(ValueError, 'non-release ref'):
            CHECKER.check(self.bundle('--all'))

    def test_requires_main(self):
        with self.assertRaisesRegex(ValueError, 'must contain main'):
            CHECKER.check(self.bundle('refs/tags/v1.0.0'))

    def test_checks_expected_main(self):
        with self.assertRaisesRegex(ValueError, 'expected snapshot'):
            CHECKER.check(self.bundle('--all'), '0' * 40)
