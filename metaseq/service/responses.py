# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import uuid
import time
from metaseq.service.constants import CHECKPOINT_LOCAL


class OAIResponse:
    def __init__(self, results: list) -> None:
        self.results = results
        self.response_id = str(uuid.uuid4())
        self.created = int(time.time())

    def __dict__(self):
        return {
            "id": self.response_id,
            "object": "text_completion",
            "created": self.created,
            "model": CHECKPOINT_LOCAL,
            "choices": [
                {
                    "text": result["text"],
                    "logprobs": {
                        "tokens": result["tokens"],
                        "token_logprobs": result["token_scores"],
                        "text_offset": result["text_offset"],
                        "top_logprobs": result["top_logprobs"],
                        "finish_reason": "length",  # TODO: implement this
                    },
                }
                for result in self.results
            ],
        }
