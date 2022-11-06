#!/bin/bash

LOCAL_CHECKPOINT_DIR=$1
TOTAL_FILES=$2
NFS_CHECKPOINT_DIR=$3
NUM_UPDATE=$4

# shellcheck disable=SC2010
NUM_COMPLETE=$(ls "${LOCAL_CHECKPOINT_DIR}"/ | grep -c "_done_checkpoint_${NUM_UPDATE}*")
while [ "$NUM_COMPLETE" -le "$TOTAL_FILES" ]
do
sleep 10
# shellcheck disable=SC2010
NUM_COMPLETE=$(ls "${LOCAL_CHECKPOINT_DIR}"/ | grep -c "_done_checkpoint_${NUM_UPDATE}*")
done

# TODO: use rsync?
cp "${LOCAL_CHECKPOINT_DIR}"/checkpoint_"${NUM_UPDATE}"* "${NFS_CHECKPOINT_DIR}"/
echo "Done copying to NFS...cleaning up."
rm "${LOCAL_CHECKPOINT_DIR}"/*checkpoint_"${NUM_UPDATE}"*