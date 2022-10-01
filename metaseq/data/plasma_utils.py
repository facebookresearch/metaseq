# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import subprocess
import tempfile

try:
    import pyarrow.plasma as plasma

    PYARROW_AVAILABLE = True
except ImportError:
    plasma = None
    PYARROW_AVAILABLE = False


class PlasmaArray:
    """
    Wrapper around numpy arrays that automatically moves the data to shared
    memory upon serialization. This is particularly helpful when passing numpy
    arrays through multiprocessing, so that data is not unnecessarily
    duplicated or pickled.
    """

    def __init__(self, array):
        super().__init__()
        self.array = array
        self.disable = array.nbytes < 134217728  # disable for arrays <128MB
        self.object_id = None
        self.path = None

        # variables with underscores shouldn't be pickled
        self._client = None
        self._server = None
        self._server_tmp = None
        self._plasma = None

    @property
    def plasma(self):
        if self._plasma is None and not self.disable:
            self._plasma = plasma
        return self._plasma

    def start_server(self):
        if self.plasma is None or self._server is not None:
            return
        assert self.object_id is None
        assert self.path is None
        self._server_tmp = tempfile.NamedTemporaryFile()
        self.path = self._server_tmp.name
        self._server = subprocess.Popen(
            ["plasma_store", "-m", str(int(1.05 * self.array.nbytes)), "-s", self.path]
        )

    @property
    def client(self):
        if self._client is None:
            assert self.path is not None
            self._client = self.plasma.connect(self.path, num_retries=200)
        return self._client

    def __getstate__(self):
        """Called on pickle load"""
        if self.plasma is None:
            return self.__dict__
        if self.object_id is None:
            self.start_server()
            self.object_id = self.client.put(self.array)
        state = self.__dict__.copy()
        del state["array"]
        state["_client"] = None
        state["_server"] = None
        state["_server_tmp"] = None
        state["_plasma"] = None
        return state

    def __setstate__(self, state):
        """Called on pickle save"""
        self.__dict__.update(state)
        if self.plasma is None:
            return
        self.array = self.client.get(self.object_id)

    def __del__(self):
        if self._server is not None:
            self._server.kill()
            self._server = None
            self._server_tmp.close()
            self._server_tmp = None


DEFAULT_PLASMA_PATH = "/tmp/plasma"
GB100 = (1024**3) * 100


class PlasmaStore:
    def __init__(self, path=DEFAULT_PLASMA_PATH, nbytes: int = GB100):

        self.server = self.start(path, nbytes)

    def __del__(self):
        self.server.kill()

    @staticmethod
    def start(path=DEFAULT_PLASMA_PATH, nbytes: int = GB100) -> subprocess.Popen:
        if not PYARROW_AVAILABLE:
            raise ImportError("please run pip install pyarrow")
        # best practice is to allocate more space than we need. The limitation seems to be the size of /dev/shm
        _server = subprocess.Popen(["plasma_store", "-m", str(nbytes), "-s", path])
        plasma.connect(path, num_retries=200)  # If we can't connect we fail immediately
        return _server
