# SPDX-License-Identifier: Apache-2.0
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from unittest.mock import patch

from .assets import configure_models
from .client import RemoteSession


def require_version():
    from importlib.metadata import PackageNotFoundError, version

    try:
        found = version('mineru')
    except PackageNotFoundError as exc:
        raise RuntimeError('install snowllm[mineru] to use snowllm-mineru') from exc
    if found != '4.1.3':
        raise RuntimeError(f'snowllm-mineru requires mineru==4.1.3; found {found}')


def install(stack, client):
    require_version()
    from mineru.config import get_config
    from mineru.model.runtime import hybrid

    if hybrid.AtomModelSingleton._models or hybrid.HybridLocalModelContextSingleton._models:
        raise RuntimeError('install the MinerU inference adapter before loading any models')
    directory = stack.enter_context(TemporaryDirectory(prefix='snowllm-mineru-models-'))
    root = Path(directory)
    for name, value in client.description['assets'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('invalid model metadata path')
        target = root / 'small-models' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)
    configure_models(stack, root, lambda path, **kw: RemoteSession(client, path, **kw),
                     device='cpu', require_weights=False)
    config = get_config()
    config.model.vlm.server_url = client.url
    config.model.vlm.model = client.model
    config.model.vlm.api_key = client.http.headers.get('Authorization', '').removeprefix('Bearer ')
    original = hybrid.atom_model_init

    def initialize(*args, **kwargs):
        name = kwargs.get('model_name', args[0] if args else None)
        if name == 'layout':
            from .layout import Layout
            return Layout(root / 'small-models/Layout/PP-DocLayoutV2', client.load)
        if name == 'mfr':
            from .mfr import Formula
            return Formula(root / 'small-models/MFR/pp_formulanet_plus_m', client.load)
        if name == 'ocr':
            from .ocr import OCR
            options = {key: value for key, value in kwargs.items()
                       if key not in ('model_name', 'small_backend', 'device')}
            return OCR(client.load, device='cpu', **options)
        return original(*args, **kwargs)

    stack.enter_context(patch.object(hybrid, 'atom_model_init', initialize))
