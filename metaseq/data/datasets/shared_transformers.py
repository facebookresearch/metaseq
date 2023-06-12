from typing import Any
import regex


def identity_transformer(x: Any) -> Any:
    return x


def remove_non_alpha_from_beginning(text: str) -> str:
    """
    Given string, remove all non-"letter" characters from beginning.
    Will remove spaces, numbers, special characters (like hyphens and colons)
    """
    while not text[0].isalpha():
        text = text[1:]

    return text


def get_first_match_of_first_capture_group(rgx: str, input_str: str) -> str:
    """
    Given a regex with a capture group and a string, it will return the first
    match in the input string.

    Note that only the first capture group is used. If no capture groups are
    present then it will raise an exception.

    :param regexp rgx: Regex string to use
    :return str: the first match in the input string.
    """
    extract_number_regex = regex.compile(rgx, flags=regex.MULTILINE | regex.DOTALL)

    matches = extract_number_regex.findall(input_str)
    assert len(matches) > 0, f"Could not find a match for r'{rgx}' in '{input_str}'"
    return matches[0]


def get_first_number(s: str) -> str:
    """
    Given string return first number surrounded by parenthesis

    Examples:

        get_first_number(' (2) something something') -> '2'
        get_first_number(' (31) something something') -> '31'
    """
    return get_first_match_of_first_capture_group(r".*?(\d+).*?", s)
