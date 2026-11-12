import argparse
from email.parser import BytesParser
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename


PLATFORMS = ('manylinux_2_28_x86_64', 'win_amd64')


def kernel_requirement(wheel):
    with zipfile.ZipFile(wheel) as archive:
        metadata, = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        values = BytesParser().parsebytes(archive.read(metadata)).get_all('Requires-Dist', [])
    required, = [Requirement(value) for value in values
                 if canonicalize_name(Requirement(value).name) == 'snowllm-kernels']
    pin, = required.specifier
    if pin.operator != '==' or '*' in pin.version or required.url or required.marker:
        raise ValueError('release must require one exact snowllm-kernels version')
    return pin.version


def verify(path, version, platform):
    name, actual, _, tags = parse_wheel_filename(path.name)
    if name != 'snowllm-kernels' or str(actual) != version or {str(tag) for tag in tags} != {'py3-none-' + platform}:
        raise ValueError('downloaded kernel wheel does not match the release dependency')
    with zipfile.ZipFile(path) as archive:
        prefix = f'snowllm_kernels-{version}.dist-info/'
        metadata = BytesParser().parsebytes(archive.read(prefix + 'METADATA'))
        if canonicalize_name(metadata['Name']) != 'snowllm-kernels' or metadata['Version'] != version:
            raise ValueError('downloaded kernel wheel metadata does not match its filename')
        windows = platform == 'win_amd64'
        for name in (('snowllm.dll', 'snowllm_loader.dll') if windows else ('libsnowllm.so', 'libsnowllm_loader.so')):
            if not archive.read('snowllm_kernels/' + name):
                raise ValueError('empty kernel wheel payload')


def check(wheel):
    version = kernel_requirement(wheel)
    with tempfile.TemporaryDirectory() as folder:
        for platform in PLATFORMS:
            destination = Path(folder) / platform
            subprocess.run([sys.executable, '-m', 'pip', 'download', '--only-binary=:all:',
                            '--no-deps', '--pre', '--platform', platform, '--python-version', '3.12',
                            '--implementation', 'py', '--abi', 'none', '--dest', str(destination),
                            'snowllm-kernels==' + version], check=True)
            artifact, = destination.glob('*.whl')
            verify(artifact, version, platform)
            print(f'release dependency: {artifact.name} PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wheel', type=Path)
    args = parser.parse_args()
    try:
        check(args.wheel)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, f'release dependency: {error}\n')
