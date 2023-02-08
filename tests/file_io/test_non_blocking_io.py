#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures
import io
import os
import shutil
import tempfile
import unittest
from typing import Optional
from unittest.mock import Mock, patch

from metaseq.file_io.common import NativePathHandler, PathManager
from metaseq.file_io.common.non_blocking_io import (
    NonBlockingBufferedIO,
    NonBlockingIO,
    NonBlockingIOManager,
)


class TestNativeIOAsync(unittest.TestCase):
    """
    This test class is meant to have comprehensive tests for
    `NativePathHandler`. Async functionality tests for other
    `PathHandler`-s should only require a single test since
    all `PathHandler`-s operate in the same way.
    """

    _tmpdir: Optional[str] = None
    _pathmgr = PathManager()

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def run(self, result=None):
        with patch("iopath.common.event_logger.EventLogger.log_event"):
            super(TestNativeIOAsync, self).run(result)

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls) -> None:
        # Cleanup temp working dir.
        if cls._tmpdir is not None:
            shutil.rmtree(cls._tmpdir)  # type: ignore

    def setUp(self) -> None:
        # Reset class variables set by methods before each test.
        self._pathmgr.set_cwd(None)
        self._pathmgr._native_path_handler._non_blocking_io_manager = None
        self._pathmgr._native_path_handler._non_blocking_io_executor = None
        self._pathmgr._async_handlers.clear()

    def test_opena(self) -> None:
        _tmpfile = os.path.join(self._tmpdir, "async.txt")
        try:
            # Write the files.
            with self._pathmgr.opena(_tmpfile + "f", "w") as f:
                f.write("f1 ")
                with self._pathmgr.opena(_tmpfile + "g", "w") as g:
                    f.write("f2 ")
                    g.write("g1 ")
                    f.write("f3 ")
                f.write("f4 ")
            with self._pathmgr.opena(_tmpfile + "f", "a") as f:
                f.write("f5 ")
            F_STR = "f1 f2 f3 f4 f5 "
            G_STR = "g1 "

            # Test that `PathManager._async_handlers` keeps track of all
            # `PathHandler`-s where `opena` is used.
            self.assertCountEqual(
                [type(handler) for handler in self._pathmgr._async_handlers],
                [type(self._pathmgr._native_path_handler)],
            )
            # Test that 2 paths were properly logged in `NonBlockingIOManager`.
            manager = self._pathmgr._native_path_handler._non_blocking_io_manager
            self.assertEqual(len(manager._path_to_data), 2)
        finally:
            # Join the threads to wait for files to be written.
            self.assertTrue(self._pathmgr.async_close())

        # Check that both files were asynchronously written and written in order.
        with self._pathmgr.open(_tmpfile + "f", "r") as f:
            self.assertEqual(f.read(), F_STR)
        with self._pathmgr.open(_tmpfile + "g", "r") as g:
            self.assertEqual(g.read(), G_STR)
        # Test that both `NonBlockingIO` objects `f` and `g` are finally closed.
        self.assertEqual(len(manager._path_to_data), 0)

    def test_async_join_behavior(self) -> None:
        _tmpfile = os.path.join(self._tmpdir, "async.txt")
        _tmpfile_contents = "Async Text"
        try:
            for _ in range(1):  # Opens 1 thread
                with self._pathmgr.opena(_tmpfile + "1", "w") as f:
                    f.write(f"{_tmpfile_contents}-1")
            for _ in range(2):  # Opens 2 threads
                with self._pathmgr.opena(_tmpfile + "2", "w") as f:
                    f.write(f"{_tmpfile_contents}-2")
            for _ in range(3):  # Opens 3 threads
                with self._pathmgr.opena(_tmpfile + "3", "w") as f:
                    f.write(f"{_tmpfile_contents}-3")
            _path_to_data = (
                self._pathmgr._native_path_handler._non_blocking_io_manager._path_to_data
            )
            # Join the threads for the 1st and 3rd file and ensure threadpool completed.
            _path_to_data_copy = dict(_path_to_data)
            self.assertTrue(
                self._pathmgr.async_join(
                    _tmpfile + "1", _tmpfile + "3"
                )  # Removes paths from `_path_to_io`.
            )
            self.assertFalse(_path_to_data_copy[_tmpfile + "1"].thread.is_alive())
            self.assertFalse(_path_to_data_copy[_tmpfile + "3"].thread.is_alive())
            self.assertEqual(len(_path_to_data), 1)  # 1 file remaining
        finally:
            # Join all the remaining threads
            _path_to_data_copy = dict(_path_to_data)
            self.assertTrue(self._pathmgr.async_close())

        # Ensure data cleaned up.
        self.assertFalse(_path_to_data_copy[_tmpfile + "2"].thread.is_alive())
        self.assertEqual(len(self._pathmgr._async_handlers), 0)
        self.assertEqual(len(_path_to_data), 0)  # 0 files remaining

    def test_opena_normpath(self) -> None:
        _filename = "async.txt"
        # `_file1` and `_file2` should represent the same path but have different
        # string representations.
        _file1 = os.path.join(self._tmpdir, _filename)
        _file2 = os.path.join(self._tmpdir, ".", _filename)
        self.assertNotEqual(_file1, _file2)
        try:
            _file1_text = "File1 text"
            _file2_text = "File2 text"
            with self._pathmgr.opena(_file1, "w") as f:
                f.write(_file1_text)
            with self._pathmgr.opena(_file2, "a") as f:
                f.write(_file2_text)
            _path_to_data = (
                self._pathmgr._native_path_handler._non_blocking_io_manager._path_to_data
            )
            # Check that `file2` is marked as the same file as `file1`.
            self.assertEqual(len(_path_to_data), 1)
            self.assertTrue(self._pathmgr.async_join())
            # Check that both file paths give the same file contents.
            with self._pathmgr.open(_file1, "r") as f:
                self.assertEqual(f.read(), _file1_text + _file2_text)
            with self._pathmgr.open(_file2, "r") as f:
                self.assertEqual(f.read(), _file1_text + _file2_text)
        finally:
            self.assertTrue(self._pathmgr.async_close())

    def test_async_consecutive_join_calls(self) -> None:
        _file = os.path.join(self._tmpdir, "async.txt")
        try:
            self.assertTrue(self._pathmgr.async_join())
            try:
                with self._pathmgr.opena(_file, "w") as f:
                    f.write("1")
            finally:
                self.assertTrue(self._pathmgr.async_join())
            with self._pathmgr.open(_file, "r") as f:
                self.assertEqual(f.read(), "1")

            try:
                f = self._pathmgr.opena(_file, "a")
                f.write("2")
                f.close()
            finally:
                self.assertTrue(self._pathmgr.async_join())
            with self._pathmgr.open(_file, "r") as f:
                self.assertEqual(f.read(), "12")
        finally:
            self.assertTrue(self._pathmgr.async_close())

    def test_opena_mode_restriction(self) -> None:
        _file = os.path.join(self._tmpdir, "async.txt")
        with self.assertRaises(ValueError):
            self._pathmgr.opena(_file, "r")
        with self.assertRaises(ValueError):
            self._pathmgr.opena(_file, "rb")
        with self.assertRaises(ValueError):
            self._pathmgr.opena(_file, "wrb")

    def test_opena_args_passed_correctly(self) -> None:
        _file = os.path.join(self._tmpdir, "async.txt")
        try:
            # Make sure that `opena` args are used correctly by using
            # different newline args.
            with self._pathmgr.opena(_file, "w", newline="\r\n") as f:
                f.write("1\n")
            with self._pathmgr.opena(_file, "a", newline="\n") as f:
                f.write("2\n3")
        finally:
            self.assertTrue(self._pathmgr.async_close())

        # Read the raw file data without converting newline endings to see
        # if the `opena` args were used correctly.
        with self._pathmgr.open(_file, "r", newline="") as f:
            self.assertEqual(f.read(), "1\r\n2\n3")

    def test_opena_with_callback(self) -> None:
        _file_tmp = os.path.join(self._tmpdir, "async.txt.tmp")
        _file = os.path.join(self._tmpdir, "async.txt")
        _data = "Asynchronously written text"

        # pyre-fixme[53]: Captured variable `_data` is not annotated.
        # pyre-fixme[53]: Captured variable `_file` is not annotated.
        # pyre-fixme[53]: Captured variable `_file_tmp` is not annotated.
        # pyre-fixme[3]: Return type must be annotated.
        def cb():
            # Insert a test to make sure `_file_tmp` was closed before
            # the callback is called.
            with open(_file_tmp, "r") as f:
                self.assertEqual(f.read(), _data)
            self._pathmgr.copy(_file_tmp, _file, overwrite=True)

        mock_cb = Mock(side_effect=cb)

        try:
            with self._pathmgr.opena(
                _file_tmp, "w", callback_after_file_close=mock_cb
            ) as f:
                f.write(_data)
        finally:
            self.assertTrue(self._pathmgr.async_close())
        # Callback should have been called exactly once.
        mock_cb.assert_called_once()

        # Data should have been written to both `_file_tmp` and `_file`.
        with open(_file_tmp, "r") as f:
            self.assertEqual(f.read(), _data)
        with open(_file, "r") as f:
            self.assertEqual(f.read(), _data)

    def test_opena_with_callback_only_called_once(self) -> None:
        _file_tmp = os.path.join(self._tmpdir, "async.txt.tmp")

        mock_cb = Mock()

        # Callback should be called once even if `close` is called
        # multiple times.
        try:
            f = self._pathmgr.opena(_file_tmp, "w", callback_after_file_close=mock_cb)
            f.close()
            f.close()
            f.close()
        finally:
            self.assertTrue(self._pathmgr.async_close())
        # Callback should have been called exactly once.
        mock_cb.assert_called_once()

    def test_async_custom_executor(self) -> None:
        # At first, neither manager nor executor are set.
        self.assertIsNone(self._pathmgr._native_path_handler._non_blocking_io_manager)
        self.assertIsNone(self._pathmgr._native_path_handler._non_blocking_io_executor)
        # Then, override the `NativePathHandler` and set a custom executor.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=128, thread_name_prefix="my prefix"
        )
        ph = NativePathHandler(async_executor=executor)
        self._pathmgr.register_handler(ph, allow_override=True)
        self.assertEqual(ph, self._pathmgr._native_path_handler)

        # Opening a file with `opena` initializes the manager with the
        # executor.
        _file = os.path.join(self._tmpdir, "async.txt")
        try:
            with self._pathmgr.opena(_file, "w") as f:
                f.write("Text")
            # Make sure the manager's executor is the same as the user's.
            self.assertEqual(
                executor,
                self._pathmgr._native_path_handler._non_blocking_io_manager._pool,
            )
        finally:
            self.assertTrue(self._pathmgr.async_close())

    def test_non_blocking_io_seekable(self) -> None:
        _file = os.path.join(self._tmpdir, "async.txt")
        # '^' marks the current position in stream

        # Test seek.
        try:
            with self._pathmgr.opena(_file, "wb") as f:
                f.write(b"012345")  # file = 012345^
                f.seek(1)  # file = 0^12345
                f.write(b"##")  # file = 0##^345
        finally:
            self.assertTrue(self._pathmgr.async_join())
            with self._pathmgr.open(_file, "rb") as f:
                self.assertEqual(f.read(), b"0##345")

        # Test truncate.
        try:
            with self._pathmgr.opena(_file, "wb") as f:
                f.write(b"012345")  # file = 012345^
                f.seek(2)  # file = 01^2345
                f.truncate()  # file = 01^
        finally:
            self.assertTrue(self._pathmgr.async_join())
            with self._pathmgr.open(_file, "rb") as f:
                self.assertEqual(f.read(), b"01")

        # Big test for seek and truncate.
        try:
            with self._pathmgr.opena(_file, "wb") as f:
                f.write(b"0123456789")  # file = 0123456789^
                f.seek(2)  # file = 01^23456789
                f.write(b"##")  # file = 01##^456789
                f.seek(3, io.SEEK_CUR)  # file = 01##456^789
                f.truncate()  # file = 01##456^
                f.write(b"$")  # file = 01##456$^
        finally:
            self.assertTrue(self._pathmgr.async_join())
            with self._pathmgr.open(_file, "rb") as f:
                self.assertEqual(f.read(), b"01##456$")

        # Test NOT tellable.
        try:
            with self._pathmgr.opena(_file, "wb") as f:
                with self.assertRaises(ValueError):
                    f.tell()
        finally:
            self._pathmgr.async_close()


class TestNonBlockingIO(unittest.TestCase):
    _tmpdir: Optional[str] = None
    _io_manager = NonBlockingIOManager(buffered=False)
    _buffered_io_manager = NonBlockingIOManager(buffered=True)

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls) -> None:
        # Cleanup temp working dir.
        if cls._tmpdir is not None:
            shutil.rmtree(cls._tmpdir)  # type: ignore

    def test_select_io(self) -> None:
        self.assertEqual(self._io_manager._IO, NonBlockingIO)
        self.assertEqual(self._buffered_io_manager._IO, NonBlockingBufferedIO)

    def test_io_manager(self) -> None:
        _file = os.path.join(self._tmpdir, "non_buffered.txt")

        try:
            # Test IO notifies manager after every write call.
            f = self._io_manager.get_non_blocking_io(
                path=_file, io_obj=open(_file, "w")
            )
            with patch.object(
                f,
                "_notify_manager",
                # pyre-fixme[16]: Item `IO` of `Union[IO[bytes], IO[str]]` has no
                #  attribute `_notify_manager`.
                wraps=f._notify_manager,
            ) as mock_notify_manager:
                f.write("." * 1)
                f.write("." * 2)
                f.write("." * 3)
                # Should notify manager 3 times: 3 write calls.
                self.assertEqual(mock_notify_manager.call_count, 3)
                mock_notify_manager.reset_mock()
                # Should notify manager 1 time: 1 close call.
                f.close()
                self.assertEqual(mock_notify_manager.call_count, 1)
        finally:
            self.assertTrue(self._io_manager._join())
            self.assertTrue(self._io_manager._close_thread_pool())

        with open(_file, "r") as f:
            self.assertEqual(f.read(), "." * 6)

    def test_buffered_io_manager(self) -> None:
        _file = os.path.join(self._tmpdir, "buffered.txt")

        try:
            # Test IO doesn't flush until buffer is full.
            f = self._buffered_io_manager.get_non_blocking_io(
                path=_file, io_obj=open(_file, "wb"), buffering=10
            )
            with patch.object(f, "flush", wraps=f.flush) as mock_flush:
                with patch.object(
                    f,
                    "_notify_manager",
                    # pyre-fixme[16]: Item `IO` of `Union[IO[bytes], IO[str]]` has
                    #  no attribute `_notify_manager`.
                    wraps=f._notify_manager,
                ) as mock_notify_manager:
                    f.write(b"." * 9)
                    mock_flush.assert_not_called()  # buffer not filled - don't flush
                    mock_notify_manager.assert_not_called()
                    # Should flush when full.
                    f.write(b"." * 13)
                    mock_flush.assert_called_once()  # buffer filled - should flush
                    # `flush` should notify manager 4 times: 3 `file.write` and 1 `buffer.close`.
                    # Buffer is split into 3 chunks of size 10, 10, and 2.
                    # pyre-fixme[16]: Item `IO` of `Union[IO[bytes], IO[str]]` has
                    #  no attribute `_buffers`.
                    self.assertEqual(len(f._buffers), 2)  # 22-byte and 0-byte buffers
                    self.assertEqual(mock_notify_manager.call_count, 4)
                    mock_notify_manager.reset_mock()
                    # `close` should notify manager 2 times: 1 `buffer.close` and 1 `file.close`.
                    f.close()
                    self.assertEqual(mock_notify_manager.call_count, 2)

            # Test IO flushes on file close.
            f = self._buffered_io_manager.get_non_blocking_io(
                path=_file, io_obj=open(_file, "ab"), buffering=10
            )
            with patch.object(f, "flush", wraps=f.flush) as mock_flush:
                f.write(b"." * 5)
                mock_flush.assert_not_called()
                f.close()
                mock_flush.assert_called()  # flush on exit
        finally:
            self.assertTrue(self._buffered_io_manager._join())
            self.assertTrue(self._buffered_io_manager._close_thread_pool())

        with open(_file, "rb") as f:
            self.assertEqual(f.read(), b"." * 27)


if __name__ == "__main__":
    unittest.main()
