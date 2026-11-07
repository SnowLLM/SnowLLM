from .._capi import check, lib
from ._common import _p, _stream, empty_bytes
from .operation import operator

PACKED_WEIGHT = 0x10000


@operator
def flash_prepare_weight(role, quant, meta, fmt, n, k):
    query = getattr(lib, 'snowllm_flash_layout_bytes', None)
    size = query(role, fmt, 1, n, k) if query is not None else 0
    if not size:
        return quant, meta, fmt
    packed_quant = empty_bytes(query(role, fmt, 0, n, k)) if role == 0 else quant
    packed_meta = empty_bytes(size)
    check(lib.snowllm_flash_prepare_weight(role, fmt, _p(quant), _p(meta),
          _p(packed_quant) if role == 0 else None, _p(packed_meta), n, k, _stream()),
          'flash_prepare_weight')
    return packed_quant, packed_meta, fmt | PACKED_WEIGHT
