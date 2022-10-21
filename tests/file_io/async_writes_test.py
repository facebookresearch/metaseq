# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
    Utility for testing async writes with `NativePathHandler`.
    Usage:
        python -m tests.file_io.async_writes_test
"""

import logging
import os
import tempfile
import time

from metaseq.file_io.common import PathManager


# pyre-fixme[5]: Global expression must be annotated.
logger = logging.getLogger(__name__)


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def printx(str):
    logger.warning(f"[{time.strftime('%X')}] {str}")


class TestDriver:
    LEN = 100000000  # This many characters per append job
    NUM_JOBS = 10
    _pathmgr = PathManager()

    def test(self) -> None:
        with tempfile.TemporaryDirectory() as _tmpdir:
            URI = os.path.join(_tmpdir, "test.txt")

            start_time = time.time()
            printx(
                f"Start dispatching {self.NUM_JOBS} async write jobs "
                f"each with {self.LEN} characters"
            )

            FINAL_STR = ""
            with self._pathmgr.opena(URI, "a") as f:
                for i in range(self.NUM_JOBS):  # `i` goes from 0 to 9
                    FINAL_STR += f"{i}" * self.LEN
                    f.write(f"{i}" * self.LEN)

            mid_time = time.time()
            printx(
                f"Time taken to dispatch {self.NUM_JOBS} threads: {mid_time - start_time}"
            )
            printx("Calling `async_join()`")
            # We want this `async_join` call to take time. If it is instantaneous, then our
            # async write calls are not running asynchronously.
            assert self._pathmgr.async_join()
            printx(
                f"Time Python waited for `async_join()` call to finish: {time.time() - mid_time}"
            )

            assert self._pathmgr.async_close()

            with self._pathmgr.open(URI, "r") as f:
                assert f.read() == FINAL_STR

            printx("Async Writes Test finish.")
            printx(
                "Passing metric: "
                "If the `async_join()` call took more than a negligible time to complete, "
                "then Python waited for the threads to finish and the Async Writes "
                "Test SUCCEEDS. Otherwise FAILURE."
            )


if __name__ == "__main__":
    printx("Async Writes Test starting.")
    tst = TestDriver()
    tst.test()
