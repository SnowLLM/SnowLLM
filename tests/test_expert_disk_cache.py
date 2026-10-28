# SPDX-License-Identifier: Apache-2.0
import concurrent.futures
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from snowllm.checkpoint import _cache_io
from snowllm.checkpoint.expert_cache import ExpertDiskCache
from snowllm.errors import SnowLLMError


def publish_in_process(roots, expert):
    cache = ExpertDiskCache(roots)
    record = cache.plan(3, 'down', 17001, 8, 'source-1')
    cache.publish(3, 'down', expert, bytes([expert]) * record)


def try_lock(path, exclusive, ready, acquired):
    fd = os.open(path, os.O_RDWR | getattr(os, 'O_BINARY', 0))
    try:
        ready.set()
        with _cache_io.locked(fd, exclusive):
            acquired.set()
    finally:
        os.close(fd)


class DiskCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.roots = [Path(self.tmp.name) / name for name in ('专家 cache a', '专家 cache b')]

    def cache(self, identity='source-1', size=17001):
        cache = ExpertDiskCache(self.roots)
        record = cache.plan(3, 'down', size, 8, identity)
        return cache, record

    def read(self, spans):
        data = bytearray(sum(s[2] for s in spans))
        for path, offset, size, dest in spans:
            with open(path, 'rb') as file:
                file.seek(offset)
                data[dest:dest + size] = file.read(size)
        return bytes(data)

    def test_stripes_are_aligned_and_reusable(self):
        cache, record = self.cache()
        self.assertFalse(any(p.exists() for p in self.roots))
        self.assertIsNone(cache.lookup(3, 'down', 4))
        data = bytes(range(251)) * (record // 251) + bytes(record % 251)
        cache.publish(3, 'down', 4, memoryview(data))
        spans = cache.lookup(3, 'down', 4)
        self.assertEqual(len(spans), 2)
        self.assertEqual(self.read(spans), data)
        self.assertTrue(all(offset % 4096 == size % 4096 == dest % 4096 == 0
                            for _, offset, size, dest in spans))
        self.assertLessEqual(abs(spans[0][2] - spans[1][2]), 4096)
        second, _ = self.cache()
        self.assertEqual(self.read(second.lookup(3, 'down', 4)), data)
        self.assertIsNone(second.lookup(3, 'down', 5))
        changed, _ = self.cache('source-2')
        self.assertIsNone(changed.lookup(3, 'down', 4))

    def test_partial_write_and_truncation_recover(self):
        cache, record = self.cache()
        data = bytes([47]) * record
        write = cache._write
        calls = [0]
        def failing(fd, data, offset):
            calls[0] += 1
            if calls[0] == 3:
                raise OSError('interrupted second disk')
            return write(fd, data, offset)
        with patch.object(cache, '_write', side_effect=failing):
            with self.assertRaises(OSError):
                cache.publish(3, 'down', 1, data)
        self.assertIsNone(cache.lookup(3, 'down', 1))
        cache.publish(3, 'down', 1, data)
        cache.publish(3, 'down', 2, data)
        path = cache.bundles[3, 'down']['shards'][1][0]
        with path.open('r+b') as file:
            file.truncate(4096)
        self.assertIsNone(cache.lookup(3, 'down', 1))
        cache.publish(3, 'down', 1, data)
        self.assertEqual(self.read(cache.lookup(3, 'down', 1)), data)
        self.assertIsNone(cache.lookup(3, 'down', 2))

    def test_concurrent_publish_and_layout_invalidation(self):
        cache, record = self.cache()
        def publish(expert):
            other, _ = self.cache()
            other.publish(3, 'down', expert, bytes([expert]) * record)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(publish, [0, 1, 2, 3, 1, 2]))
        for expert in range(4):
            self.assertEqual(self.read(cache.lookup(3, 'down', expert)), bytes([expert]) * record)
        changed, _ = self.cache(size=17002)
        self.assertIsNone(changed.lookup(3, 'down', 1))

    def test_layout_version_invalidates_same_sized_records(self):
        from snowllm.checkpoint import expert_cache
        cache, record = self.cache()
        cache.publish(3, 'down', 1, bytes([19]) * record)
        with patch.object(expert_cache, 'LAYOUT_VERSION', expert_cache.LAYOUT_VERSION + 1):
            changed, _ = self.cache()
            self.assertIsNone(changed.lookup(3, 'down', 1))
            self.assertNotEqual(changed.bundles[3, 'down']['index'], cache.bundles[3, 'down']['index'])
        self.assertEqual(self.read(cache.lookup(3, 'down', 1)), bytes([19]) * record)

    def test_environment_validation(self):
        with patch.dict(os.environ, {'SNOWLLM_MOE_CACHE_DIRS': ''}):
            self.assertIsNone(ExpertDiskCache.from_env())
        root = str(self.roots[0])
        for value in [os.pathsep + root, root + os.pathsep,
                      os.pathsep.join([root, str(self.roots[0] / '.')])]:
            with patch.dict(os.environ, {'SNOWLLM_MOE_CACHE_DIRS': value}):
                with self.assertRaises(SnowLLMError):
                    ExpertDiskCache.from_env()
        with patch.dict(os.environ, {'SNOWLLM_MOE_CACHE_DIRS': os.pathsep.join(map(str, self.roots))}):
            self.assertEqual(ExpertDiskCache.from_env().directories, tuple(self.roots))

    def test_multiple_processes_publish_without_losing_records(self):
        ctx = multiprocessing.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=3, mp_context=ctx) as pool:
            list(pool.map(publish_in_process, [self.roots] * 6, [0, 1, 2, 3, 1, 2]))
        cache, record = self.cache()
        for expert in range(4):
            self.assertEqual(self.read(cache.lookup(3, 'down', expert)), bytes([expert]) * record)

    def test_shared_readers_and_exclusive_writer(self):
        cache, _ = self.cache()
        bundle = cache.bundles[3, 'down']
        with cache._locked(bundle, True):
            pass
        ctx = multiprocessing.get_context('spawn')
        for exclusive, blocked in [(False, False), (True, True)]:
            ready, acquired = ctx.Event(), ctx.Event()
            process = ctx.Process(target=try_lock, args=(bundle['index'], exclusive, ready, acquired))
            try:
                with cache._locked(bundle, False):
                    process.start()
                    self.assertTrue(ready.wait(15), 'child did not open the index')
                    self.assertEqual(acquired.wait(.3 if blocked else 15), not blocked)
                self.assertTrue(acquired.wait(15), 'lock was not released')
                process.join(15)
                self.assertEqual(process.exitcode, 0)
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join(15)

    def test_single_directory_and_short_writes(self):
        cache = ExpertDiskCache(self.roots[:1])
        record = cache.plan(3, 'down', 17001, 8, 'source-1')
        data = (bytes(range(256)) * (record // 256))
        write = _cache_io.pwrite
        with patch.object(_cache_io, 'pwrite', side_effect=lambda fd, data, offset:
                          write(fd, data[:97], offset)):
            cache.publish(3, 'down', 7, data)
        spans = cache.lookup(3, 'down', 7)
        self.assertEqual(len(spans), 1)
        self.assertEqual(self.read(spans), data)

class DiskCacheImportTests(unittest.TestCase):
    def test_import_and_planning_do_not_require_unix_locking(self):
        code = '''
import os
import sys
sys.modules['fcntl'] = None
os.environ.pop('SNOWLLM_MOE_CACHE_DIRS', None)
from snowllm.checkpoint.expert_cache import ExpertDiskCache
assert ExpertDiskCache.from_env() is None
os.environ['SNOWLLM_MOE_CACHE_DIRS'] = 'expert-cache'
cache = ExpertDiskCache.from_env()
assert cache.plan(3, 'down', 17, 8, 'source-1') == 4096
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1], timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
