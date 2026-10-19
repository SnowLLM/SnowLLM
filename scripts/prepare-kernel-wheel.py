import ast
from email.parser import BytesParser
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

try:
    from packaging.requirements import Requirement
    from packaging.tags import sys_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename


def prepare(public_wheel, kernel_dist, stage, root):
    with zipfile.ZipFile(public_wheel) as archive:
        metadata, = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        requirements = BytesParser().parsebytes(archive.read(metadata)).get_all('Requires-Dist', [])
    requirement, = [Requirement(value) for value in requirements
                    if canonicalize_name(Requirement(value).name) == 'snowllm-kernels']
    pin, = requirement.specifier
    if pin.operator != '==' or '*' in pin.version or requirement.marker or requirement.url:
        raise ValueError('snowllm-kernels must have one exact version requirement')
    compatible = set(sys_tags())
    candidates = []
    for path in sorted(kernel_dist.glob('snowllm_kernels-*.whl')):
        name, version, _, tags = parse_wheel_filename(path.name)
        if name == 'snowllm-kernels' and str(version) == pin.version and tags & compatible:
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f'expected one compatible snowllm-kernels=={pin.version} wheel in '
                         f'{kernel_dist}, found {len(candidates)}; '
                         'set SNOWLLM_KERNELS_DIST to the verified release directory')
    tree = ast.parse((root / 'snowllm/_capi.py').read_text())
    abi, = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'ABI_VERSION'
                    for target in node.targets)]
    checker = root.parent / 'SnowLLM-Kernels/scripts/check-wheel.py'
    if not checker.is_file():
        raise ValueError('private wheel checker is required; use --python-only for a standalone Python wheel')
    candidate = stage / candidates[0].name
    shutil.copyfile(candidates[0], candidate)
    subprocess.run([sys.executable, str(checker), str(candidate), '--version', pin.version,
                    '--abi', str(abi)], check=True)


if __name__ == '__main__':
    try:
        prepare(*(Path(value) for value in sys.argv[1:]))
    except (OSError, ValueError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        sys.exit(f'kernel wheel: {error}')
