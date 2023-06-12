from typing import Any, Dict

from metaseq.data.datasets import openai_generated_transformers
from metaseq.data.datasets.shared_transformers import get_first_number
from metaseq.data.datasets.types import OAITeacherGeneratedDatasetItem, OAITeacherGeneratedDatasetItemLogprobs


def _adjust_teacher_generated_format(data: Dict) -> OAITeacherGeneratedDatasetItem:
    """
    The format that we have for the teacher generated data for hellaswag is not
    what we expect as OpenAI output, so this function will transform the data
    the correct shape.
    """
    raw_response = data["response"]

    # move human label from range [0,3] to [1,4] so we match with the
    # download_hellaswag script
    human_label = int(data.get("label", "-2"))
    human_label += 1

    return {
        "source": data["prompt"],
        "human": str(human_label),
        "text": raw_response["text"],
        "finish_reason": raw_response["finish_reason"],
        "index": data["ind"],
        "logprobs": raw_response["logprobs"],
    }


def hellaswag_before_transforming_into_metaseq_inference(raw_dict: Any) -> OAITeacherGeneratedDatasetItem:
    # Transform data to the correct OpenAI output shape
    item = _adjust_teacher_generated_format(raw_dict)

    # remove everything after EOS
    item = openai_generated_transformers.remove_all_tokens_after_eos_sanitizer(item)

    # replace EOS with </s>
    item = openai_generated_transformers.replace_eos_sanitizer(item, eos_replacement="</s>")

    # if found, remove everything after the token that has a closing bracket in
    # it. This regex will match any token that has a closing bracked in it. For
    # examples:
    #    - ") "
    #    - ")"
    item = openai_generated_transformers.truncate_after_token(item, r".*?\).*?")

    # verify that the target text contains a number. This will throw if not
    # found and item will be skipped
    try:
        get_first_number(item["text"])
    except AssertionError:
        raise ValueError(f"Could not find a number in the generated text: {item['text']}")

    return item


def hellaswag_convert_model_output_domain_to_original_domain(model_output: str) -> str:
    # example model_output:
    #   ' (4) something something'
    number_s = get_first_number(model_output)
    choice_idx = number_s.strip()

    # model generated label is in range [1,4]
    return choice_idx
