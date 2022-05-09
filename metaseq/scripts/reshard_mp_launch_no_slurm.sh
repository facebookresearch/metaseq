#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

prefix=$1
save_dir=$2
mparts=$3
tgt_size=$4
shift 4
mkdir -p $save_dir
let "last_part=$mparts -1"
echo "$@"
for i in $(seq 0 $last_part)
do

  echo "python -m metaseq.scripts.reshard_mp $prefix $save_dir --part $i --target-ddp-size $tgt_size"
  jname=reshard_mp"$i"_ddp"$tgt_size"
  echo $jname
  python3 -m metaseq.scripts.reshard_mp $prefix $save_dir --part $i --target-ddp-size $tgt_size &
done
echo "Waiting on jobs..."
wait $(jobs -p)
echo "Done"

