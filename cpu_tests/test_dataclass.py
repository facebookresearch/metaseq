# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from metaseq.dataclass.configs import DynamicConfig

class TestDynamicConfig(unittest.TestCase):
    def test_malformed_config_load(self):
        metaseq_dir = pathlib.Path(__file__).parent.parent.resolve()
        malformed_config = os.path.join(metaseq_dir, "config/malformed.json")
        dcfg = DynamicConfig(
            json_file_path=malformed_config
        )

    def test_empty_config_load(self):
        metaseq_dir = pathlib.Path(__file__).parent.parent.resolve()
        empty_config = os.path.join(metaseq_dir, "config/empty.json")
        dcfg = DynamicConfig(
            json_file_path=empty_config
        )

    def test_nonexistent_config_load(self):
        nonexistent_config = "404.json"
        dcfg = DynamicConfig(
            json_file_path=nonexistent_config
        )
        

if __name__ == "__main__":
    unittest.main()
