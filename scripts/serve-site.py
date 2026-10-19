#!/usr/bin/env python3
"""Preview the assembled site, including VitePress clean URLs."""
import argparse
from functools import partial
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent


class SiteHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        api = getattr(self.server, 'status_api_base', None)
        if api and urlsplit(self.path).path in ('/status/', '/status/index.html'):
            page = (ROOT / 'web/status/index.html').read_text().replace(
                'data-api-base="https://status.snowllm.dev"', f'data-api-base="{escape(api, quote=True)}"')
            body = page.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            return BytesIO(body)
        return super().send_head()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, max-age=0')
        super().end_headers()

    def translate_path(self, path):
        resolved = Path(super().translate_path(path))
        if not resolved.exists() and not resolved.suffix:
            html = resolved.with_suffix('.html')
            if html.is_file():
                return str(html)
        return str(resolved)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=4173)
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--status-api-base', help='Override the status backend origin for this preview only')
    args = parser.parse_args()
    if args.status_api_base:
        origin = urlsplit(args.status_api_base)
        if (origin.scheme not in ('http', 'https') or not origin.hostname or origin.username
                or origin.password or origin.path not in ('', '/') or origin.query or origin.fragment):
            parser.error('--status-api-base must be an HTTP(S) origin')
    server = ThreadingHTTPServer((args.bind, args.port), partial(SiteHandler, directory=str(ROOT / 'web')))
    server.status_api_base = args.status_api_base
    print(f'SnowLLM preview: http://localhost:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
