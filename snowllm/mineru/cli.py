# SPDX-License-Identifier: Apache-2.0
import argparse
import contextlib
import os
import sys
from unittest.mock import patch

import httpx


COMMANDS = ('parse', 'api-server', 'file-parse-server', 'webui')


def patch_webui(stack, url, model):
    from mineru.kit.gradio.client import ManagedLocalApiServer

    original = ManagedLocalApiServer._command

    def command(self, port, upload_dir):
        native = original(self, port, upload_dir)
        if native[:4] != [sys.executable, '-m', 'mineru.kit.main', 'api-server']:
            raise RuntimeError('unsupported MinerU managed API launcher')
        prefix = [sys.executable, '-m', 'snowllm.mineru.cli', '--snowllm-url', url]
        if model:
            prefix += ['--snowllm-model', model]
        return [*prefix, *native[3:]]

    stack.enter_context(patch.object(ManagedLocalApiServer, '_command', command))


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--snowllm-url', default=os.getenv('SNOWLLM_MINERU_URL', 'http://127.0.0.1:8000'))
    parser.add_argument('--snowllm-model', default=os.getenv('SNOWLLM_MINERU_MODEL'))
    args, forwarded = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    from .injection import install, require_version

    try:
        require_version()
        from mineru.kit.main import main as native_main

        command = next((value for value in forwarded if not value.startswith('-')), None)
        help_only = any(value in forwarded for value in ('--help', '-h', '--version'))
        if command == 'router' and not help_only:
            raise RuntimeError('use mineru-kit router with --upstream-url pointing to snowllm-mineru api-server')
        if command == 'vlm-server' and not help_only:
            raise RuntimeError('start the model service with snowllm b/mineru')
        with contextlib.ExitStack() as stack:
            if not help_only and command in COMMANDS:
                stack.enter_context(patch.dict(os.environ, {'MINERU_DEVICE_MODE': 'cpu'}))
                if command == 'webui':
                    patch_webui(stack, args.snowllm_url, args.snowllm_model)
                else:
                    from .client import ModelClient

                    client = ModelClient(args.snowllm_url, args.snowllm_model,
                                         api_key=os.getenv('SNOWLLM_MINERU_API_KEY'))
                    stack.callback(client.close)
                    install(stack, client)
            stack.enter_context(patch.object(sys, 'argv', ['snowllm-mineru', *forwarded]))
            return native_main()
    except (ImportError, RuntimeError, ValueError, httpx.HTTPError) as exc:
        parser.exit(1, f'snowllm-mineru: {exc}\n')


if __name__ == '__main__':
    main()
