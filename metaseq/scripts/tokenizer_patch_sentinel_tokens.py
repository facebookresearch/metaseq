from tokenize import Special
from tokenizers import (
    Tokenizer,
    models,
)
import json
from pprint import pprint

NUM_IMAGE_TOKENS = 2**13
NUM_SENTINEL_TOKENS = 512 + 2

# self.cm3_sentinel_end_ind 65537
# self.cm3_break_ind 65536
# self.cm3_sentinel_end = "<eoss>"
# self.cm3_break = "<racm3:break>"


def create_reserved_sentinel_tokens():
    all_tokens = ["<racm3:break>", "<eoss>"]
    all_tokens += [f"<sentinel:{i}>" for i in range(NUM_SENTINEL_TOKENS)]
    return all_tokens


import argparse

parser = argparse.ArgumentParser(
    description="Augment tokenizer to include image tokens safely"
)

parser.add_argument(
    "--input-path",
    type=str,
    required=False,
    default="/data/cm3z/armenag/mmlm/tokenizers/gpt2-unified-image.json",
    help="Filepath for the input tokenizer",
)
parser.add_argument(
    "--output-path",
    type=str,
    required=False,
    default="/data/cm3z/liliyu/mmlm/tokenizers/gpt2-unified-image-racm3-patch.json",
    help="Filepath for the output tokenizer file",
)


args = parser.parse_args()

tokenizer_state = json.load(open(args.input_path))
tokenizer_state["pre_tokenizer"]["pretokenizers"].insert(
    0,
    {
        "type": "Split",
        # "pattern": {"Regex": "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z"},
        "pattern": {"Regex": "<sentinel:[0-9]+>"},  # r"<sentinel:[0-9]+>"
        "behavior": "Isolated",
        "invert": False,
    },
)

vocab_offset = 65536
sentinels = create_reserved_sentinel_tokens()
for i in range(len(sentinels)):
    ind = vocab_offset + i
    token = sentinels[i]
    tokenizer_state["model"]["vocab"][token] = ind
    # tokenizer_state["added_tokens"][ind]["content"] = token
    tokenizer_state["added_tokens"].append(
        {
            "id": ind,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
    )


tokenizer_updated = Tokenizer(models.BPE()).from_str(json.dumps(tokenizer_state))
pprint(tokenizer_updated.encode("<sentinel:0><sentinel:1><racm3:break><eoss>").tokens)
pprint(tokenizer_updated.encode("<sentinel:0><sentinel:1><racm3:break><eoss>").ids)

tokenizer_updated.save(args.output_path)
