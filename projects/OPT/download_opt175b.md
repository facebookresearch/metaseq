# Accessing OPT-175B

After receiving an email with a presigned URL to access the model weights, follow the below set of instructions to get started with hosting the model.

## Download all shards
Since we trained OPT-175B on 124 hosts, we have 124\*8 = 992 files corresponding to the model parameters (8x tensor parallelism) which will take approximately 328GB of available storage. The presigned URL that you receive in your email will look something like the following:

```
https://<cloudfront_url>/175b/checkpoint_last_20220503/stubbed.pt?&<super_long_query_string>
```

To download all 992 files, run:
```
bash metaseq/scripts/download_opt175b.sh "<presigned_url_given_in_email>"
```

Make sure to wrap the url in quotes here.  You will get a 403 error otherwise.

By default this will download the files into the directory that the command is run in. You can also optionally include a target directory by running:

```
bash metaseq/scripts/download_opt175b.sh "<presigned_url_given_in_email>" -d "<target directory>"
```
If the download is interrupted, you can continue the download from the latest available file by adding -c flag to the bash script.

### md5sum check
In some cases, files may be corrupted after downloading.  To confirm this is not the case, check the [md5sum of your downloaded files](./assets/opt175b_md5sum_shards.csv).

To check your files, you can run:
```bash
md5sum *
```

## Reshard the shards
To consolidate the 992 shards into 8 files model-parallel evaluation, run the `metaseq.scripts.reshard_fsdp` script:
```bash
for j in {0..7}; do
    python -m metaseq.scripts.reshard_fsdp \
    --input-glob-pattern "/path/to/raw/checkpoints/checkpoint_last-model_part-$j-shard*.pt" \
    --output-shard-name "/path/to/resharded/checkpoints/reshard-model_part-$j.pt" \
    --num-output-shards 1 --skip-optimizer-state True --unflatten-weights True
done
```
Note that most of our models expect to run with Model (Tensor) Parallelism. For smaller models, some
users may find it easier to eliminate model parallelism. The checkpoints can be converted
to eliminate use of MP with the `consolidate_fsdp_shards.py` script:

```bash
python metaseq.scripts.consolidate_fsdp_shards ${FOLDER_PATH}/checkpoint_last --new-arch-name transformer_lm_gpt --save-prefix ${FOLDER_PATH}/consolidated
```

### md5sum check
Once you have consolidated the shards, you should have the following checksums:
```
934def0c596e01dfb849fa65b73e01aa  dict.txt
75a37753dd7a28a2c5df80c28bf06e4e  gpt2-merges.txt
cf410ee085c5c69c957bb1f6d8456596  gpt2-vocab.json
7e71cb65c4be784aa0b2889ac6039ee8  reshard-model_part-0.pt
c8123da04f2c25a9026ea3224d5d5022  reshard-model_part-1.pt
45e5d10896382e5bc4a7064fcafd2b1e  reshard-model_part-2.pt
abb7296c4d2fc17420b84ca74fc3ce64  reshard-model_part-3.pt
05dcc7ac6046f4d3f90b3d1068e6da15  reshard-model_part-4.pt
d24dd334019060ce1ee7e625fcf6b4bd  reshard-model_part-5.pt
fb1615ce0bbe89cc717f3e5079ee2655  reshard-model_part-6.pt
2f3124432d2dbc6aebfca06be4b791c2  reshard-model_part-7.pt
```


## Run the API
Follow the instructions in the [API docs](../../docs/api.md) to spin up the API.  You will need to update the constants in `metaseq/service/constants.py` to point to right directories.

Note that the `gpt2-merges.txt` and `gpt2-vocab.json` files in [`projects/OPT/assets/`](/projects/OPT/assets) will need to be moved to the corresponding directories defined in the `constants.py` file. You can directly download them with:

```bash
cd /path/to/resharded-weights
wget https://github.com/facebookresearch/metaseq/raw/main/projects/OPT/assets/gpt2-merges.txt
wget https://github.com/facebookresearch/metaseq/raw/main/projects/OPT/assets/gpt2-vocab.json
```
