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
bash metaseq/scripts/download_opt175b.sh "<presigned_url_given_in_email>" "<target directory>"
```

## Reshard the shards
To consolidate the 992 shards into 8 files model-parallel evaluation, run (assuming you have SLURM set up already):
```
bash metaseq/scripts/reshard_mp_launch.sh <directory_where_all_the_shards_are>/checkpoint_last <output_dir>/ 8 1
```
If you don't have slurm on your machine, run:
```
bash metaseq/scripts/reshard_mp_launch_no_slurm.sh <directory_where_all_the_shards_are>/checkpoint_last <output_dir>/ 8 1
```

Note that most of our models expect to run with Model (Tensor) Parallelism. For smaller models, some
users may find it easier to eliminate model parallelism. The checkpoints can be converted
to eliminate use of MP with the `consolidate_fsdp_shards.py` script:

```bash
python metaseq.scripts.consolidate_fsdp_shards ${FOLDER_PATH}/checkpoint_last --new-arch-name transformer_lm_gpt --save-prefix ${FOLDER_PATH}/consolidated
```


## Run the API
Follow the instructions in the [API docs](../../docs/api.md) to spin up the API.  You will need to update the constants in `metaseq/service/constants.py` to point to right directories.

Note that the `gpt2-merges.txt` and `gpt2-vocab.json` files in [`projects/OPT/assets/`](/projects/OPT/assets) will need to be moved to the corresponding directories defined in the `constants.py` file. You can directly download them with:

```bash
cd /path/to/resharded-weights
wget https://github.com/facebookresearch/metaseq/raw/main/projects/OPT/assets/gpt2-merges.txt
wget https://github.com/facebookresearch/metaseq/raw/main/projects/OPT/assets/gpt2-vocab.json
```
