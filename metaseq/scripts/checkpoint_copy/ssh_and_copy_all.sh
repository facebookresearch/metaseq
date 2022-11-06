#!/bin/bash
OSS_DEST=$1
LOCAL_CHECKPOINT_DIR=$2
TOTAL_FILES=$3
NFS_CHECKPOINT_DIR=$4

for HOST in `scontrol show hostnames $SLURM_JOB_NODELIST` ; do
    ssh $HOST "$OSS_DEST/metaseq/scripts/checkpoint_copy/poll_and_copy_single.sh $LOCAL_CHECKPOINT_DIR $TOTAL_FILES $NFS_CHECKPOINT_DIR"
done
