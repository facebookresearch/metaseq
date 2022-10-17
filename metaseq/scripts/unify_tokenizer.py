from tokenizers import ByteLevelBPETokenizer
from fire import Fire
import os


def main(path):
    merges_file = os.path.join(path, "gpt2-merges.txt")
    vocab_file = os.path.join(path, "gpt2-vocab.json")
    unified_file = os.path.join(path, "gpt2-unified.json")
    if os.path.exists(unified_file):
        return
    if not (os.path.exists(merges_file and os.path.exists(vocab_file))):
        raise ValueError(
            f"Path {path} is missing one or both of {merges_file} and {vocab_file}"
        )

    old_tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
    old_tokenizer.save(unified_file)


if __name__ == "__main__":
    Fire(main)
