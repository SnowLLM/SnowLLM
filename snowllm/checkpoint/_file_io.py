# SPDX-License-Identifier: Apache-2.0
import os


def pread(fd, size, offset):
    if os.name != 'nt':
        return os.pread(fd, size, offset)
    # Seek-based reads require exclusive access to the descriptor.
    os.lseek(fd, offset, os.SEEK_SET)
    return os.read(fd, size)


def pwrite(fd, data, offset):
    if os.name != 'nt':
        return os.pwrite(fd, data, offset)
    os.lseek(fd, offset, os.SEEK_SET)
    return os.write(fd, data)
