from tokenizers import ByteLevelBPETokenizer
from fire import Fire


def main(vocab_file, merges_file, unified_path):
    old_tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
    old_tokenizer.save(unified_path)


if __name__ == "__main__":
    Fire(main)
