# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

from snowllm.trace_archive import archive_trace, export_profile


class TraceArchiveTests(unittest.TestCase):
    def test_export_produces_one_json_member_and_no_loose_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dict(traceEvents=[dict(name='kernel', ts=1.5, dur=2.25)])
            profile = Mock()
            profile.export_chrome_trace.side_effect = lambda path: Path(path).write_text(json.dumps(data))
            for _ in range(2):
                export_profile(profile, root)
            archives = list(root.glob('*.pt.trace.tar.gz'))
            self.assertEqual(len(archives), 2)
            self.assertEqual(len(list(root.iterdir())), 2)
            for path in archives:
                with tarfile.open(path) as archive:
                    members = archive.getmembers()
                    self.assertEqual(len(members), 1)
                    self.assertTrue(members[0].name.endswith('.pt.trace.json'))
                    self.assertEqual(Path(members[0].name).name, members[0].name)
                    self.assertEqual(json.load(archive.extractfile(members[0])), data)

    def test_failed_archive_preserves_exported_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.json'
            path.write_text('{"traceEvents":[]}')
            with patch('snowllm.trace_archive.tarfile.open', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    archive_trace(path)
            self.assertEqual(list(path.parent.iterdir()), [path])
            self.assertEqual(json.loads(path.read_text()), dict(traceEvents=[]))


if __name__ == '__main__':
    unittest.main()
