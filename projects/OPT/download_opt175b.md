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
    --input "/path/to/raw/checkpoints/checkpoint_last-model_part-$j-shard*.pt" \
    --output "/path/to/resharded/checkpoints/reshard-model_part-$j.pt" \
    --num-output-shards 1 --skip-optimizer-state True --unflatten-weights True
done
```
Note that most of our models expect to run with Model (Tensor) Parallelism. For smaller models, some
users may find it easier to eliminate model parallelism. The checkpoints can be converted
to eliminate use of MP with the `reshard_mp.py` script:

```bash
python -m metaseq.scripts.reshard_mp \
    --input "/path/to/resharded/checkpoints/reshard-model_part-*.pt" \
    --output "/path/to/mp/resharded/checkpoints/reshard-model_part-{i}.pt" \
    --num-output-parts 1
```

### md5sum check
Once you have consolidated the FSDP shards, you should have the following checksums with the options `--skip-optimizer-state True` and `--unflatten-weights True`:
```
934def0c596e01dfb849fa65b73e01aa  dict.txt
75a37753dd7a28a2c5df80c28bf06e4e  gpt2-merges.txt
cf410ee085c5c69c957bb1f6d8456596  gpt2-vocab.json
14284cd62d00166bf4a16b7ac43b4946  reshard-model_part-0.pt
bd9daba744d65f628e24a561ebc692ae  reshard-model_part-1.pt
8e3b35dfd55ad6dc6340458ad7c7c3a1  reshard-model_part-2.pt
48f1f189b33976392e7e64677382c6b7  reshard-model_part-3.pt
9e5906c44cef3f4996aec6f02fe01bef  reshard-model_part-4.pt
32b69d07dce2dd6b38f987a09a3286dd  reshard-model_part-5.pt
e9b9de69f5373d1435a8b2e9a3093d7e  reshard-model_part-6.pt
99349ce86ed401f715759dca68c35358  reshard-model_part-7.pt
```


## Run the API
Follow the instructions in the [API docs](../../docs/api.md) to spin up the API.  You will need to update the constants in `metaseq/service/constants.py` to point to right directories.

Note that the `gpt2-merges.txt` and `gpt2-vocab.json` files in [`projects/OPT/assets/`](/projects/OPT/assets) will need to be moved to the corresponding directories defined in the `constants.py` file. You can directly download them with:

```bash
cd /path/to/resharded-weights
wget https://github.com/facebookresearch/metaseq/raw/main/projects/OPT/assets/gpt2-merges.txt
wget https://github.com/facebookresearch/metaseq/raw/main/projects/OPT/assets/gpt2-vocab.json
```
