# SPDX-License-Identifier: Apache-2.0
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from snowllm.hub import pi


class PiTests(unittest.TestCase):
    def test_configuration_uses_running_model_context(self):
        recipe = SimpleNamespace(serve_as='local', dir='local', sources=[], defaults={})
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root)/'models.json'
            response = io.BytesIO(json.dumps({'data': [dict(id='local', context_length=491520)]}).encode())
            with patch.object(pi, 'MODELS_JSON', str(dest)), \
                 patch.object(pi.recipes, 'catalogue', return_value=[recipe]), \
                 patch.object(pi.urllib.request, 'urlopen', return_value=response):
                self.assertEqual(pi.main('pi', []), 0)
            config = json.loads(dest.read_text())
            self.assertEqual(config['providers']['snowllm']['models'][0]['contextWindow'], 491520)

    def test_missing_runtime_context_does_not_write_guessed_capacity(self):
        recipe = SimpleNamespace(serve_as='local', dir='local', sources=[], defaults={})
        for data in ({'data': []}, {'data': [dict(id='local')]},
                     {'data': [dict(id='local', context_length=True)]}):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as root:
                dest = Path(root)/'models.json'
                dest.write_text('{"existing": true}')
                with patch.object(pi, 'MODELS_JSON', str(dest)), \
                     patch.object(pi.recipes, 'catalogue', return_value=[recipe]), \
                     patch.object(pi.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(data).encode())):
                    self.assertEqual(pi.main('pi', []), 1)
                self.assertEqual(json.loads(dest.read_text()), {'existing': True})


if __name__ == '__main__':
    unittest.main()
