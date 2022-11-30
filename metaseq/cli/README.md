# Easily running custom metaseq models

Say you're running one of the interactive inference pragroms in `metaseq.cli`.
You probably needed to make changes to `metaseq.service.constants`: perhaps you changed the `LAUNCH_ARGS` or `MODEL_PARALLEL` setting, or your overrode the `CHECKPOINT_FOLDER`.

This gets inconvenient, since you may be experimenting with multiple models, and you'd like to be able to pull changes without overwriting your modified `constants.py`.

## Using overrides

1. Copy your overridden `constants.py` to another folder (for example, `~/my_constants/my_first_override.py`) [1].
2. Export an environment variable so that the above folder is in your `PYTHONPATH`: `export PYTHONPATH=$PYTHONPATH:~/my_constants`
3. Export an environment variable with your override's module name: `export METASEQ_SERVICE_CONSTANTS_MODULE=my_first_override`

Now, when you run an interactive inferencer from this directory, it'll use your overridden constants.

### [1] Example contents of `my_first_override.py`

Note: this is modified from the main branch as of the time of writing.
Comments highlight changes.

```
import os

MAX_SEQ_LEN = 1024                                                # <--- changed
BATCH_SIZE = 2048
MAX_BATCH_TOKENS = 3072
DEFAULT_PORT = 6010
MODEL_PARALLEL = 2                                                # <--- changed
TOTAL_WORLD_SIZE = 8
MAX_BEAM = 32

CHECKPOINT_FOLDER = "/shared/my_cool_checkpoint"                  # <--- changed

# tokenizer files
BPE_MERGES = os.path.join(CHECKPOINT_FOLDER, "gpt2-merges.txt")
BPE_VOCAB = os.path.join(CHECKPOINT_FOLDER, "gpt2-vocab.json")
MODEL_FILE = os.path.join(CHECKPOINT_FOLDER, "reshard.pt")

LAUNCH_ARGS = [
    f"--model-parallel-size {MODEL_PARALLEL}",
    f"--distributed-world-size {TOTAL_WORLD_SIZE}",
    "--ddp-backend fully_sharded",                                # <--- changed
    "--task language_modeling",
    f"--bpe-merges {BPE_MERGES}",
    f"--bpe-vocab {BPE_VOCAB}",
    "--bpe hf_byte_bpe",
    f"--merges-filename {BPE_MERGES}",
    f"--vocab-filename {BPE_VOCAB}",
    f"--path {MODEL_FILE}",
    "--beam 1",
    "--distributed-port 13000",
    "--checkpoint-shard-count 1",
    f"--batch-size {BATCH_SIZE}",
    f"--buffer-size {BATCH_SIZE * MAX_SEQ_LEN}",
    f"--max-tokens {BATCH_SIZE * MAX_SEQ_LEN}",
    "/tmp",  # required "data" argument.
]
```

## Suggested serving

You can create an override for each of the models you'd like to be able to use, and you can put them all in the same folder.
Then, when you'd like to swap to use a different model, just change the environment variable we exported in step 3 above and relaunch.
