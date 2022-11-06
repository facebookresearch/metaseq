#!/bin/bash

LOCAL_CHECKPOINT_DIR=$1
TOTAL_FILES=$2
NFS_CHECKPOINT_DIR=$3

# shellcheck disable=SC2012
NUM_COMPLETE=$(ls '_done*' "${LOCAL_CHECKPOINT_DIR}"/ | wc -l)
while [ "$NUM_COMPLETE" -le "$TOTAL_FILES" ]
do
sleep 10
# shellcheck disable=SC2012
NUM_COMPLETE=$(ls '_done*' "${LOCAL_CHECKPOINT_DIR}"/ | wc -l)
done

# TODO: use rsync?
cp "${LOCAL_CHECKPOINT_DIR}"/checkpoint* "${NFS_CHECKPOINT_DIR}"/
