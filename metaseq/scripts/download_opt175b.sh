#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# presigned_url format: "https://<redacted>.cloudfront.net/175b/checkpoint_last_20220503/stubbed.pt?<redacted>"
# filename format: checkpoint_last-model_part-[0-7]-shard[0-123].pt

# To download all the of the parameters for the 175B model, run:
# bash ./download_opt175b.sh "<presigned_url_given_in_email>"

presigned_url=$1
str_to_replace='stubbed.pt'
for part_id in $(seq 0 7)
do
  for shard_id in $(seq 0 123)
  do
    filename="checkpoint_last-model_part-$part_id-shard$shard_id.pt"
    wget -O "$filename" "${presigned_url/$str_to_replace/$filename}"
  done
done


