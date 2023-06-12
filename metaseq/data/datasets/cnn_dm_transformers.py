from typing import Any, Dict

from metaseq.data.datasets import openai_generated_transformers, shared_transformers
from metaseq.data.datasets.types import OAITeacherGeneratedDatasetItem


def before_transforming_into_metaseq_inference(raw_dict: Any) -> OAITeacherGeneratedDatasetItem:
    item: OAITeacherGeneratedDatasetItem = raw_dict

    item = openai_generated_transformers.remove_all_tokens_after_eos_sanitizer(item)
    item = openai_generated_transformers.replace_eos_sanitizer(item, eos_replacement="</s>")

    return item


def convert_teacher_domain_to_original_domain(model_output: str) -> str:
    """Convert output from text-davinci-003 to the format of original dataset"""

    original_domain_output = shared_transformers.remove_non_alpha_from_beginning(model_output)

    return original_domain_output
