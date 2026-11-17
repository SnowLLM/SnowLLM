# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import socket
import tarfile
import tempfile
import time


def archive_trace(path: Path) -> Path:
    output = path.with_suffix('.tar.gz')
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + output.name,
                                     suffix='.tmp', delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        with tarfile.open(pending, 'w:gz') as archive:
            archive.add(path, arcname=path.name)
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
    path.unlink()
    return output


def export_profile(profile, directory: Path) -> None:
    path = directory / f'{socket.gethostname()}_{os.getpid()}.{time.time_ns()}.pt.trace.json'
    profile.export_chrome_trace(str(path))
    archive_trace(path)
