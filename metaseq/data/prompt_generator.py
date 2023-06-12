from argparse import ArgumentError
import json
import random
import math
from jinja2 import Environment, meta

from collections import defaultdict
from glob import glob
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from typing import Generator, List, Optional

from metaseq.logging import get_logger

logger = get_logger(__name__)

prompt_template_default = """
{%- for shot in shots -%}
{{ shot['src'] }} {{ shot['tgt'] }}

{%+ endfor -%}
{{ input_s }}"""


class PromptGenerator:
    """
    This class is responsible for decorating the current sample that is intended
    to be used as the input to the LM with additional information. The goal of
    adding this additional information is to help the LM better understand the
    context of the current sample.

    See :meth:`get_next_prompt` for more details on how the prompt is decorated
    and what decorations are possible.
    """

    def __init__(
        self,
        prompt_template_s: Optional[str] = None,
        n_fewshot_samples: int = 0,
        few_shot_data_glob_path: Optional[str] = None,
        fewshot_sample_method='random',
        rng_seed=42,
        max_tokens: int = None,
        shots_to_input_truncation_percentage: float = 0.5,
        tokenizer_vocab_file_path="/mnt/input_data_dir/pretrained_models/OPT/dependencies/gpt2-vocab.json",
        tokenizer_merges_file_path="/mnt/input_data_dir/pretrained_models/OPT/dependencies/gpt2-merges.txt",
    ):
        """
        Initializes the PromptGenerator

        :param Optional[str] prompt_template_s: A Jinja template string
            Must contain `input_s`
            If `n_fewshot_samples` > 0, must contain `shots` loop
            Each shot should be a dictionary with `src` and `tgt` keys
        :param Optional[str] few_shot_data_glob_path: If provided, this will be the
            path to the few-shot data that will be used to generate the prompts.
            If None, no few-shot data will be used. Defaults to None.
        :param int n_fewshot_samples: The number of few-shot samples that we
            want to add to the prompt. If this is > 0 then `few_shot_data_path`
            must also be provided. Defaults to 0
        :param str fewshot_sample_method: The method used to choose samples for the few shots.
            This may be random or fixed. If fixed the generator will choose first N samples from
            the given few_shot_data_path
        :param int rng_seed: Seed for the random number generator, defaults to
            42
        :param int max_tokens: The maximum tokens the prompt can use. Defaults to None.
            We are not using the exact tokenizer that Open AI uses and thus we force output LESS tokens than the actual value by
             subtracting an extra 1% of the max_tokens given to ensure result will be within the allowed range by the true tokenizer.
        :param int shots_to_input_truncation_percentage: The ratio or percentage of tokens to truncation from the shot samples vs the input sample
        :param str tokenizer_vocab_file_path: Path to tokenizer vocabulary file, defaults to "/mnt/input_data_dir/pretrained_models/OPT/dependencies/gpt2-vocab.json"
        :param str tokenizer_merges_file_path: Path to tokenizer merges file, defaults to "/mnt/input_data_dir/pretrained_models/OPT/dependencies/gpt2-merges.txt"
        """

        if prompt_template_s is None:
            prompt_template_s = prompt_template_default

        logger.debug(f'Using prompt template:\n{prompt_template_s}')
        jinja_env = Environment()
        prompt_template_ast = jinja_env.parse(prompt_template_s)
        template_variables = meta.find_undeclared_variables(prompt_template_ast)

        assert 'input_s' in template_variables, (
            f"You provided the PromptGenerator with template that did not contain required variable 'input_s'. "
            f"Please update the template and try again.\n"
            f"Template:\n{prompt_template_s}"
        )
        self._n_fewshot_samples = n_fewshot_samples
        if n_fewshot_samples > 0:
            assert 'shots' in template_variables, (
                f"You configured the PromptGenerator to use {n_fewshot_samples} shot samples; however, "
                f"the template provided do not contain the required variable 'shots'. "
                f"Please update the template and try again.\n"
                f"Template:\n{prompt_template_s}"
            )

        self._prompt_template = jinja_env.from_string(prompt_template_s)

        if few_shot_data_glob_path is not None:
            assert n_fewshot_samples > 0, "A path for the few shot data was provided but n_fewshot_samples is not a positive number"
            self._fewshot_data_generator = self._create_fewshot_sample_generator(
                few_shot_data_glob_path, fewshot_sample_method
            )

        self._rng = random.Random(rng_seed)

        assert 0 <= shots_to_input_truncation_percentage <= 1, "Parameter shots_to_input_truncation_percentage must be withing range [0, 1]"
        self._shots_to_input_truncation_percentage = shots_to_input_truncation_percentage

        self._adjusted_max_tokens = None
        self._tokenizer = None
        # This value is not correct, but we can't compute tokens without tokenizer
        self._num_tokens_of_fixed_prompt = 0

        if max_tokens is not None:
            # We are not using the exact tokenizer that Open AI uses and thus it make output LESS tokens than the actual value
            # We subtract extra 1% from the max value to ensure result will be within the allowed range by the true tokenizer
            tokenizer_gap = math.ceil(0.01 * max_tokens)
            self._adjusted_max_tokens = max_tokens - tokenizer_gap
            logger.debug(f'Maximum prompt tokens = {max_tokens} - (1% * {max_tokens})')
            logger.debug(f'                      = {max_tokens} - {tokenizer_gap}')
            logger.debug(f'                      = {self._adjusted_max_tokens}')
            self._tokenizer: Tokenizer = ByteLevelBPETokenizer.from_file(
                tokenizer_vocab_file_path,
                tokenizer_merges_file_path,
            )

            empty_shots = [{'src': '', 'tgt': ''} for shot in range(n_fewshot_samples)]
            fixed_str_portion_of_prompt = self._prompt_builder(shot_samples=empty_shots, input_s='')
            logger.debug(f'Empty {self._n_fewshot_samples}-shot template\n{fixed_str_portion_of_prompt}')
            self._num_tokens_of_fixed_prompt = len(self._tokenizer.encode(fixed_str_portion_of_prompt))
            logger.debug(f'Num tokens for fixed portion of prompt = {self._num_tokens_of_fixed_prompt}')

            assert self._adjusted_max_tokens > self._num_tokens_of_fixed_prompt, (
                f'You provided the PromptGenerator with template that contributes more minimum tokens {self._num_tokens_of_fixed_prompt} than the maximum allowed tokens {self._adjusted_max_tokens}.'
                f'Please increase the max tokens number or reduce the prompt parameters.'
            )

    def get_next_prompt(self, input_s: str) -> str:
        """
        This function takes the current sample that is being used as the input
        to the LM and decorates it according to the :class:`PromptGenerator`'s
        configuration.

        These are the possible `decorations` that this method might do to the
        input `input_s`

        - addition of `self._n_fewshot_samples` few-shot samples to the prompt

        Note that the PromptGenerator will return the `input_s` input
        as-is if it is not configured to `decorate` the prompt in any way.

        :param str input_s: current sample that is the input to the LM.
            This `base prompt` will be `decorated` by the PromptGenerator to
            create the final prompt.
        :return str: the final decorated prompt that will be used as the input
            to the LM.
        """

        # decorate the prompt with few-shot samples if needed
        samples_for_this_shot = []

        if self._n_fewshot_samples > 0:
            samples_for_this_shot = next(self._fewshot_data_generator)
            assert len(
                samples_for_this_shot
            ) == self._n_fewshot_samples, f"Fewshot data generator returned the wrong number of samples, expected {self._n_fewshot_samples} but got {len(samples_for_this_shot)}"

        prompt = self._prompt_builder(samples_for_this_shot, input_s)

        if self._adjusted_max_tokens is not None and self._tokenizer:
            prompt_token_ids = self._tokenizer.encode(prompt).ids
            prompt_token_ids_len = len(prompt_token_ids)
            logger.debug(f'Generated prompt is {prompt_token_ids_len} tokens.')

            # If the prompt tokens exceeds the maximum then truncate to fit
            if prompt_token_ids_len >= self._adjusted_max_tokens:
                logger.warn(
                    f'Generated prompt has {prompt_token_ids_len} but maximum allowed is {self._adjusted_max_tokens}. Prompt shots and input will truncated.'
                )

                num_tokens_to_truncate = (prompt_token_ids_len - self._adjusted_max_tokens) + self._num_tokens_of_fixed_prompt
                original_prompt = prompt
                original_prompt_token_ids_len = prompt_token_ids_len
                truncated_samples = []

                if self._n_fewshot_samples > 0:
                    num_shot_tokens_to_truncate = math.ceil(
                        num_tokens_to_truncate * self._shots_to_input_truncation_percentage
                    )
                    num_tokens_to_truncate_from_each_shot = math.ceil(num_shot_tokens_to_truncate / self._n_fewshot_samples)
                    num_input_tokens_to_truncate = num_tokens_to_truncate - num_shot_tokens_to_truncate

                    for sample_index, sample in enumerate(samples_for_this_shot):

                        src_token_ids = self._tokenizer.encode(sample['src']).ids
                        tgt_token_ids = self._tokenizer.encode(sample['tgt']).ids
                        src_token_percentage = len(src_token_ids) / (len(src_token_ids) + len(tgt_token_ids))
                        tgt_token_percentage = len(tgt_token_ids) / (len(src_token_ids) + len(tgt_token_ids))

                        src_tokens_to_truncate = math.ceil(num_tokens_to_truncate_from_each_shot * src_token_percentage)
                        tgt_tokens_to_truncate = math.ceil(num_tokens_to_truncate_from_each_shot * tgt_token_percentage)

                        logger.debug(
                            f'Sample {sample_index} truncation params:\n'
                            f'Tokens to truncate: {num_tokens_to_truncate_from_each_shot}\n'
                            f'Source tokens: {len(src_token_ids)} ({src_token_percentage:.2f}%)\n'
                            f'Right truncating {src_tokens_to_truncate} from source.\n'
                            f'Target tokens: {len(tgt_token_ids)} ({tgt_token_percentage:.2f}%)\n'
                            f'Right truncating {tgt_tokens_to_truncate} from target.'
                        )

                        truncated_sample = {
                            'src': self._tokenizer.decode(src_token_ids[:-src_tokens_to_truncate]),
                            'tgt': self._tokenizer.decode(tgt_token_ids[:-tgt_tokens_to_truncate])
                        }
                        truncated_samples.append(truncated_sample)
                else:
                    # Since there are 0 shots, all the truncation occurs on the input string
                    num_input_tokens_to_truncate = num_tokens_to_truncate

                logger.debug(f'Right truncating {num_input_tokens_to_truncate} from input.')
                truncated_input_s = self._truncate_n_tokens(input_s, num_input_tokens_to_truncate)
                prompt = self._prompt_builder(truncated_samples, truncated_input_s)

                logger.debug(f'Original Prompt (Tokens: {original_prompt_token_ids_len}):\n{original_prompt}\n')
                prompt_token_ids = self._tokenizer.encode(prompt).ids
                prompt_token_ids_len = len(prompt_token_ids)
                logger.debug(f'Truncated Prompt (Tokens: {prompt_token_ids_len})')

        return prompt

    def _truncate_n_tokens(self, s: str, num_tokens_to_truncate: int) -> str:
        """Given string, encode, truncate, decode."""
        return self._tokenizer.decode(self._tokenizer.encode(s).ids[:-num_tokens_to_truncate])

    def _prompt_builder(self, shot_samples: List[dict], input_s: str) -> str:
        return self._prompt_template.render(shots=shot_samples, input_s=input_s)

    def _create_fewshot_sample_generator(self, data_glob_path: str, sample_method: str) -> Generator[List[dict], None, None]:
        """
        Creates a generator that yields a list of few-shot samples. The samples
        themselves are chosen randomly from the few-shot data. Each "iteration"
        of the generator returned by this function will yield a list of
        `self._n_fewshot_samples` samples.

        :param str data_path: Path to the directly which contains the few-shot
            data. These must be `jsonl` documents.
        :yield Generator[List[dict], None, None]: Infinite generator that yields
            a list of random few-shot samples for each iteration. The size of
            the list is `self._n_fewshot_samples`.
        """
        logger.info(f"Creating few-shot sample generator from {data_glob_path}")

        file_paths = glob(data_glob_path)
        assert len(file_paths) > 0, f"No files found in {data_glob_path}"

        logger.info(f"Found the following files for creation of few-shot samples: {file_paths}")

        # get index of every newline in each file
        file_idx_to_prompt_start_idx = defaultdict(list)
        file_idx_to_handle = {}
        for file_idx, file_path in enumerate(file_paths):
            # NOTE: this approach prevents us from having to load the entire
            # file into memory but it does mean that we need to have a file
            # handle open for each file
            file_handle = open(file_path, "r")
            file_idx_to_handle[file_idx] = file_handle

            # prompt_offset is the index of the first character of the prompt
            # within the file. Initially it is 0 since the first prompt starts
            # at the beginning of the file
            prompt_offset = 0

            file_idx_to_prompt_start_idx[file_idx].append(prompt_offset)

            line = file_handle.readline()
            while line != "":
                prompt_offset += len(line)
                file_idx_to_prompt_start_idx[file_idx].append(prompt_offset)
                line = file_handle.readline()

            # pop the last entry since it happens just before EOF (there is no sample after it)
            file_idx_to_prompt_start_idx[file_idx].pop()

        # The generator state has been set up, now we only need to yield the
        # samples when needed
        while True:
            # Every time we need a new sample, we randomly choose a file and a
            # sample from that file
            chosen_file_idx = self._rng.sample(file_idx_to_prompt_start_idx.keys(), 1)[0]
            file_handle = file_idx_to_handle[chosen_file_idx]

            if sample_method == 'random':
                # Sampling self.n_fewshot_samples samples at a time ensures there
                # are no duplicates for the samples of the current prompt
                chosen_offsets = self._rng.sample(file_idx_to_prompt_start_idx[chosen_file_idx], self._n_fewshot_samples)
            elif sample_method == 'fixed':
                chosen_offsets = [file_idx_to_prompt_start_idx[chosen_file_idx][i] for i in range(self._n_fewshot_samples)]
            else:
                raise ArgumentError(
                    f"You passed an unknown sample method: {sample_method}. Allowed values are: 'random' or 'fixed'"
                )

            samples = [None] * self._n_fewshot_samples

            for i, offset in enumerate(chosen_offsets):
                file_handle.seek(offset)
                line = file_handle.readline()

                sample = json.loads(line)
                assert set(sample.keys()).issuperset({"src", "tgt"}), "jsonl entry must contain 'src' and 'tgt' fields"

                samples[i] = sample

            yield samples  #type: ignore
