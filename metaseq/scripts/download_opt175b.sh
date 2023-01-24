#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# presigned_url format: "https://<redacted>.cloudfront.net/175b/checkpoint_last_20220503/stubbed.pt?<redacted>"
# filename format: checkpoint_last-model_part-[0-7]-shard[0-123].pt

# To download all the of the parameters for the 175B model, run:
# bash ./download_opt175b.sh "<presigned_url_given_in_email>" -d "<target directory>"

# To continue the 175B model download, run:
# bash ./download_opt175b.sh "<presigned_url_given_in_email>" -d "<target directory>" -c

presigned_url=$1
str_to_replace="stubbed.pt"
continue_download=false

if [ -z $presigned_url ]; then
  echo "enter a valid presigned_url value"
  exit
elif [ $presigned_url != "-h" ]; then
  shift 1
fi

while getopts ":d:ch" option; do
   case $option in
      h)
        echo "-d /path/ for the target directory"
        echo "-c continue download"
        exit;;
      c) # download checkpoint
         continue_download=true;;
      d) # target directory
         direct=$OPTARG;;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done

for part_id in $(seq 0 7)
do
  for shard_id in $(seq 0 123)
  do
      filename="checkpoint_last-model_part-$part_id-shard$shard_id.pt"

      if [ $continue_download == false ] || [ ! -s $direct$filename ]; then
        wget -O "$direct$filename" "${presigned_url/$str_to_replace/$filename}"
      else
        echo "- ${filename} exist"
      fi
  done
done