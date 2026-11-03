# SPDX-License-Identifier: Apache-2.0
import io
import math
import zipfile


MAX_BYTES = 256 << 20


def pack(values):
    import numpy as np

    arrays = {name: np.asarray(value) for name, value in values.items()}
    if any(value.dtype.kind not in 'fibu' or value.dtype.itemsize > 8 for value in arrays.values()):
        raise ValueError('unsupported tensor dtype')
    if sum(value.nbytes for value in arrays.values()) > MAX_BYTES:
        raise ValueError('tensor payload is too large')
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def unpack(payload):
    import numpy as np

    if len(payload) > MAX_BYTES + (1 << 20):
        raise ValueError('tensor payload is too large')
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            items = archive.infolist()
            if (not items or len(items) > 32 or len({item.filename for item in items}) != len(items)
                    or any(not item.filename.endswith('.npy') for item in items)
                    or sum(item.file_size for item in items) > MAX_BYTES):
                raise ValueError('invalid tensor archive')
            for item in items:
                with archive.open(item) as stream:
                    version = np.lib.format.read_magic(stream)
                    reader = {(1, 0): np.lib.format.read_array_header_1_0,
                              (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:
                        raise ValueError('unsupported tensor header')
                    shape, _, dtype = reader(stream)
                    if (len(shape) > 8 or dtype.kind not in 'fibu' or dtype.itemsize > 8
                            or stream.tell() + math.prod(shape) * dtype.itemsize != item.file_size):
                        raise ValueError('invalid tensor shape or dtype')
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            values = {name: archive[name].copy() for name in archive.files}
        if any(value.dtype.kind not in 'fibu' or value.dtype.itemsize > 8 for value in values.values()):
            raise ValueError('unsupported tensor dtype')
        return values
    except (OSError, EOFError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError('invalid tensor payload') from exc
