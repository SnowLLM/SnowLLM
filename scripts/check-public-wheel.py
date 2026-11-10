import argparse
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


SENSITIVE = {
    'credential': rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|'
                  rb'hf_[A-Za-z0-9]{25,}|sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}|'
                  rb'(?:AKIA|ASIA)[A-Z0-9]{16}|'
                  rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----)',
    'personal path': rb'(?:/home/[^/\s]+/|/Users/[^/\s]+/|[A-Za-z]:\\Users\\[^\\\s]+\\)',
    'credential URL': rb'https?://[^\s/:@]{2,}:[^\s/@]{4,}@',
}


def check(wheel, root=None):
    match = re.fullmatch(r'snowllm-(.+)-py3-none-any\.whl', wheel.name)
    if match is None:
        raise ValueError('expected a snowllm py3-none-any wheel')
    version = match[1]
    metadata_prefix = f'snowllm-{version}.dist-info/'
    metadata_files = {'METADATA', 'WHEEL', 'RECORD', 'entry_points.txt', 'top_level.txt',
                      'LICENSE', 'licenses/LICENSE'}
    with zipfile.ZipFile(wheel) as archive:
        entries = archive.infolist()
        if len(entries) != len({entry.filename for entry in entries}):
            raise ValueError('duplicate wheel entries')
        for entry in entries:
            name = entry.filename
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
                raise ValueError('unsafe wheel entry path')
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError('symlink in public wheel')
            if name.startswith(metadata_prefix):
                allowed = name[len(metadata_prefix):] in metadata_files
            else:
                allowed = (name.startswith('snowllm/') and name.endswith('.py') or
                           name.startswith('snowllm/hub/bundle/') and name.endswith('.json') or
                           name.startswith('snowllm/daemon/static/') and path.suffix in {
                               '.html', '.css', '.js', '.svg', '.json'} or
                           name == 'snowllm/models/deepseek_v4/LICENSE-DeepSeek.txt')
                if any(part in {'internal', 'experiments', 'scratchpad', '__pycache__', '.git',
                                'csrc', 'include', 'tests'} for part in path.parts):
                    allowed = False
            if not allowed:
                raise ValueError(f'unexpected public wheel entry: {name}')
            if root and name.endswith('.py') and not (root / name).is_file():
                raise ValueError(f'wheel module is absent from source: {name}')
            blob = archive.read(name)
            for category, pattern in SENSITIVE.items():
                if re.search(pattern, blob):
                    raise ValueError(f'{category} detected in {name}; value withheld')
        metadata = BytesParser().parsebytes(archive.read(metadata_prefix + 'METADATA'))
        if metadata['Name'] != 'snowllm' or metadata['Version'] != version:
            raise ValueError('wheel metadata does not match its filename')
        tags = BytesParser().parsebytes(archive.read(metadata_prefix + 'WHEEL'))
        if tags['Root-Is-Purelib'] != 'true' or tags.get_all('Tag') != ['py3-none-any']:
            raise ValueError('wheel metadata is not pure Python')
    print(f'public wheel: {wheel.name}, {len(entries)} allowlisted files, content checks PASS')
    return len(entries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wheel', type=Path)
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    try:
        check(args.wheel, args.root)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f'public wheel: {error}\n')
