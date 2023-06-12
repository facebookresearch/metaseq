from typing import Any, Dict

from metaseq.data.datasets import openai_generated_transformers
from metaseq.data.datasets.types import OAITeacherGeneratedDatasetItem


def before_transforming_into_metaseq_inference(raw_dict: Any) -> OAITeacherGeneratedDatasetItem:
    item: OAITeacherGeneratedDatasetItem = raw_dict

    item = openai_generated_transformers.sanitize_beginning(item)
    item = openai_generated_transformers.remove_all_tokens_after_eos_sanitizer(item)
    item = openai_generated_transformers.replace_eos_sanitizer(item, eos_replacement="</s>")

    return item
