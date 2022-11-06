#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

MAIN_SLURM_NODELIST=$1
OSS_DEST=$2
LOCAL_CHECKPOINT_DIR=$3
TOTAL_FILES=$4
NFS_CHECKPOINT_DIR=$5
NUM_UPDATE=$6

echo $(klist)
echo "RUNNING ssh_and_copy_all for update $NUM_UPDATE...."

# Create checkpoint dir on NFS for this update
mkdir -p "${NFS_CHECKPOINT_DIR}${NUM_UPDATE}"

for HOST in $(scontrol show hostnames "$MAIN_SLURM_NODELIST") ; do
    ssh $HOST "$OSS_DEST/metaseq/scripts/checkpoint_copy/poll_and_copy_single.sh $LOCAL_CHECKPOINT_DIR $TOTAL_FILES $NFS_CHECKPOINT_DIR $NUM_UPDATE"
done
