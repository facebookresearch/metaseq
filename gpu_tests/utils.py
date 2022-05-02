# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utils for gpu_tests.
"""
import os
import unittest


def is_this_circleci():
    """
    Return if we are currently running in CircleCI.
    """
    return bool(os.environ.get("CIRCLECI"))


def skipIfCircleCI(testfn, reason="Test disabled in CircleCI"):
    """
    Decorate a test to skip if running on CircleCI.
    """
    return unittest.skipIf(is_this_circleci(), reason)(testfn)
