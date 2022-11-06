#!/bin/bash

MAIN_SLURM_NODELIST=$1
OSS_DEST=$2
LOCAL_CHECKPOINT_DIR=$3
TOTAL_FILES=$4
NFS_CHECKPOINT_DIR=$5

klist;

for HOST in `scontrol show hostnames $MAIN_SLURM_NODELIST` ; do
    ssh $HOST "$OSS_DEST/metaseq/scripts/checkpoint_copy/poll_and_copy_single.sh $LOCAL_CHECKPOINT_DIR $TOTAL_FILES $NFS_CHECKPOINT_DIR"
done
