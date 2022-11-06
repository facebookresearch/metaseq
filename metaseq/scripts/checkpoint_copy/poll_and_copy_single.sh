#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

LOCAL_CHECKPOINT_DIR=$1
TOTAL_FILES=$2
NFS_CHECKPOINT_DIR=$3
NUM_UPDATE=$4

# shellcheck disable=SC2010
NUM_COMPLETE=$(ls "${LOCAL_CHECKPOINT_DIR}" | grep -c "_done_checkpoint_${NUM_UPDATE}*")
while [ "$NUM_COMPLETE" -lt "$TOTAL_FILES" ]
do
  echo "Found only $NUM_COMPLETE files so far..."
  sleep 10
  # shellcheck disable=SC2010
  NUM_COMPLETE=$(ls "${LOCAL_CHECKPOINT_DIR}" | grep -c "_done_checkpoint_${NUM_UPDATE}*")
done

echo "Found $NUM_COMPLETE files! Proceeding to copy..."

# TODO: use rsync?
cp "${LOCAL_CHECKPOINT_DIR}/checkpoint_${NUM_UPDATE}*" "${NFS_CHECKPOINT_DIR}${NUM_UPDATE}"
echo "Done copying to NFS...cleaning up."
rm "${LOCAL_CHECKPOINT_DIR}/*checkpoint_${NUM_UPDATE}*"