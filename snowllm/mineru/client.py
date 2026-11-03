# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace

import httpx

from .wire import pack, unpack


class ModelClient:
    def __init__(self, url, model=None, *, api_key=None, timeout=600, transport=None):
        self.url = url.rstrip('/')
        if self.url.endswith('/v1'):
            self.url = self.url[:-3]
        self.params = {'model': model} if model else {}
        self.http = httpx.Client(timeout=timeout, transport=transport,
                                 headers={'Authorization': f'Bearer {api_key}'} if api_key else {})
        try:
            response = self.http.get(self.url + '/v1/inference', params=self.params)
            response.raise_for_status()
            self.description = response.json()
            if self.description.get('protocol') != 1:
                raise ValueError('unsupported SnowLLM auxiliary inference protocol')
            self.model = self.description['model']
            self.params = {'model': self.model}
            self.components = self.description['components']
        except BaseException:
            self.close()
            raise

    def close(self):
        self.http.close()

    def infer(self, component, values):
        response = self.http.post(self.url + '/v1/inference/' + component,
                                  params=self.params, content=pack(values),
                                  headers={'Content-Type': 'application/x-npz'})
        response.raise_for_status()
        return unpack(response.content)

    def load(self, path, architecture='PPOCRModel', **options):
        name = Path(path).name
        matches = [component for component, spec in self.components.items()
                   if Path(spec['checkpoint']).name == name and spec['architecture'] == architecture]
        if len(matches) != 1:
            raise ValueError(f'unsupported remote checkpoint: {name}')
        component = matches[0]
        spec = self.components[component]
        if any(spec['options'].get(key, False) != value for key, value in options.items() if key != 'config'):
            raise ValueError(f'remote model options do not match {component}')
        config = options.get('config', {})
        return RemoteModel(self, component, config=config)


class RemoteModel:
    def __init__(self, client, component, *, config=None):
        self.client, self.component = client, component
        self.spec = client.components[component]
        self.config = config or {}
        if 'head_channels' in self.spec:
            self.head = SimpleNamespace(weight=SimpleNamespace(shape=(self.spec['head_channels'],)))

    def to(self, device):
        if str(device) != 'cpu':
            raise ValueError('remote model adapters only accept CPU tensors')
        return self

    def eval(self):
        return self

    def __call__(self, value=None, *, pixel_values=None):
        import numpy as np
        import torch

        if value is None:
            value = pixel_values
        values = {'input': value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value}
        if self.component == 'layout' and 'class_thresholds' in self.config:
            values['class_thresholds'] = np.asarray(self.config['class_thresholds'], dtype=np.float32)
        result = self.client.infer(self.component, values)
        result = {name: bool(value) if value.ndim == 0 and value.dtype.kind == 'b'
                  else torch.from_numpy(value) for name, value in result.items()}
        if self.spec['kind'] == 'namespace':
            return SimpleNamespace(**result)
        if self.spec['kind'] == 'dict':
            return result
        ordered = tuple(result[name] for name in self.spec['outputs'])
        return ordered if self.spec['kind'] == 'tuple' else ordered[0]


class RemoteSession:
    def __init__(self, client, path, **options):
        matches = [name for name, spec in client.components.items()
                   if Path(spec['checkpoint']).name == Path(path).name]
        if len(matches) != 1 or not matches[0].startswith('table_'):
            raise ValueError(f'unsupported remote table model: {Path(path).name}')
        self.client, self.component = client, matches[0]
        self.spec = client.components[self.component]

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in self.spec['inputs']]

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in self.spec['outputs']]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self.spec['metadata'])

    def get_providers(self):
        return ['SnowLLMRemoteExecutionProvider']

    def run(self, output_names, input_feed, run_options=None):
        if set(input_feed) != set(self.spec['inputs']):
            raise ValueError('remote table input names do not match the model')
        result = self.client.infer(self.component, {'input': input_feed[self.spec['inputs'][0]]})
        return [result[name] for name in output_names or self.spec['outputs']]
