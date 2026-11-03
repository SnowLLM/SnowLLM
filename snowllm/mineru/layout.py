# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import patch

from mineru.model.layout.pp_doclayoutv2 import (
    PPDocLayoutV2ForObjectDetection,
    PPDocLayoutV2LayoutModel,
)


class Layout(PPDocLayoutV2LayoutModel):
    def __init__(self, path, load_model):
        def load(directory, **options):
            return load_model(Path(directory) / 'model.safetensors', 'PPDocLayoutV2Model',
                              config=options['config'].to_dict())

        with patch.object(PPDocLayoutV2ForObjectDetection, 'from_pretrained', load):
            super().__init__(str(path), device='cpu')
