from typing import List

import regex

from metaseq.data.datasets.shared_transformers import remove_non_alpha_from_beginning
from metaseq.data.datasets.types import OAITeacherGeneratedDatasetItem


def sanitize_beginning(data: OAITeacherGeneratedDatasetItem) -> OAITeacherGeneratedDatasetItem:
    """
    This function will remove all non-letter characters from the beginning of
    the text and the tokens list.

    Also note that this removes any non-alpha character/tokens from the
    beginning of the text, which is not always desired.
    """

    data["text"] = remove_non_alpha_from_beginning(data["text"])

    # also remove non-letter tokens from the beginning of the tokens list
    logprobs_dict = data["logprobs"]
    token_list: List[str] = logprobs_dict["tokens"]

    first_valid_idx = 0
    while not token_list[first_valid_idx].strip().isalpha():
        first_valid_idx += 1

    for key in [
        "tokens",
        "token_logprobs",
        "top_logprobs",
        "text_offset",
    ]:
        logprobs_dict[key] = logprobs_dict[key][first_valid_idx:]

    return data


def remove_all_tokens_after_eos_sanitizer(
    data: OAITeacherGeneratedDatasetItem, eos_token_name="<|endoftext|>"
) -> OAITeacherGeneratedDatasetItem:
    """
    This function will remove all tokens after the first EOS token.

    :param str eos_token_name: The name of the EOS token, defaults to
        "<|endoftext|>"
    """

    # it can be that there are some samples whose last token is not
    # "<|endoftext|>". According to conversation with Subho here [1] we should
    # remove all tokens after the endoftext
    #
    # [1]:
    #     https://teams.microsoft.com/l/message/19:ZYlNWDJ8jxO0FSqvmpwH1-sCI7RjTudf408_odtYMCU1@thread.tacv2/1677880904286?tenantId=72f988bf-86f1-41af-91ab-2d7cd011db47&groupId=72b4c54c-a4e8-4f3e-b2c3-2bbeaf09e0ff&parentMessageId=1677718389179&teamName=Distillery&channelName=General&createdTime=1677880904286&allowXTenantAccess=false
    logprobs_dict = data["logprobs"]
    token_list: List[str] = logprobs_dict["tokens"]

    # sanity check
    assert eos_token_name in token_list

    eos_index = token_list.index(eos_token_name)

    # remove everything after this index (even the eos token)
    for key in [
        "tokens",
        "token_logprobs",
        "top_logprobs",
        "text_offset",
    ]:
        logprobs_dict[key] = logprobs_dict[key][:eos_index]

    return data


def replace_eos_sanitizer(
    data: OAITeacherGeneratedDatasetItem,
    eos_replacement: str = "</s>",
    eos_token_name="<|endoftext|>"
) -> OAITeacherGeneratedDatasetItem:
    """
    This function will replace the EOS token name with the given replacement
    string.

    :param str eos_replacement: New name for the EOS token we want to use,
        defaults to "</s>"
    :param str eos_token_name: Old name that was being used for the EOS token,
        defaults to "<|endoftext|>"
    """

    logprobs_dict = data["logprobs"]

    tokens = logprobs_dict["tokens"]
    for t_idx in range(len(tokens)):
        if tokens[t_idx] == eos_token_name:
            tokens[t_idx] = eos_replacement

    top_logprobs = logprobs_dict["top_logprobs"]
    for logprob_dict in top_logprobs:
        if eos_token_name in logprob_dict:
            # remove existing item and assign it to eos_replacement token
            logprob_dict[eos_replacement] = logprob_dict.pop(eos_token_name)

    return data


def truncate_after_token(data: OAITeacherGeneratedDatasetItem, rgx: str) -> OAITeacherGeneratedDatasetItem:
    """
    This function will truncate the text and tokens list AFTER the first token
    that matches the given regex.

    :param str rgx: The regex to match the token after which we should truncate
    """
    token_matcher = regex.compile(rgx, flags=regex.MULTILINE | regex.DOTALL)

    logprobs_dict = data["logprobs"]
    token_list: List[str] = logprobs_dict["tokens"]

    # find the first token that matches the regex
    index_of_last_token = 0
    seen_text = ""
    for token in token_list:
        if token_matcher.match(token):
            break
        index_of_last_token += 1
        seen_text += token

    # if we processed all tokens and exceeded the length of the list then we
    # didn't find any token that matches the regex, so we raise an error
    if index_of_last_token == len(token_list):
        raise ValueError(f"Could not find any token that matches the regex {rgx}.")

    # right now we're at the index of the token that matched the regex,
    # meaning we want to drop everything after this index, so we add the
    # last token to seen text and then increment the index
    seen_text += token_list[index_of_last_token]
    index_of_token_after_last = index_of_last_token + 1

    for key in [
        "tokens",
        "token_logprobs",
        "top_logprobs",
        "text_offset",
    ]:
        logprobs_dict[key] = logprobs_dict[key][:index_of_token_after_last]

    # now we need to truncate the text as well
    data["text"] = seen_text

    return data
