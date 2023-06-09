import argparse
import json
import logging
import os
import pathlib
import shutil
import sys
import time
import operator
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from glob import glob
from typing import Callable, Dict, List, Optional

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from metaseq import options, utils
from metaseq.data import JsonlDataset, data_utils
from metaseq.data.prompt_generator import PromptGenerator
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as distributed_utils
from metaseq.generation_metrics.metrics import GenerationMetrics, evaluate_inference_files
from metaseq.hub_utils import GeneratorInterface
from metaseq.logging import get_logger
from metaseq.utils import flatten_config

logger = get_logger(__name__)

TOTAL_TOKENS_GENERATED = 0


def _tokenize_one_json(
    json: Dict[str, str],
    encode_fn: Callable[[str], List[int]],
    prompt_generator: PromptGenerator,
    prompt_eos_text: str,
    delimiter=None,
    max_seq_len: int = 2048,
    max_prompt_len: int = 0,
    truncation: str = "right",
    prompt_end_text: Optional[str] = None,
):
    """
    Inference Tokenization Function for Json Input

    :param Dict[str, str] json: The json line containing the prompt and target
    :param Callable encode_fn: The function to use to encode the prompt and
        target. Encodes a string to a list of token ids.
    :param PromptGenerator prompt_generator: The prompt generator to use
    :param str prompt_eos_text: The text to append at the end of prompt
    :param str delimiter: The delimiter to use to split the prompt and target
    :param int max_seq_len: The maximum sequence length
    :param int max_prompt_len: The maximum prompt length
    :param Optional[str] prompt_end_text: If provided and the prompt
        doesn't end with the specified text before being given to the model then
        the last tokens of the prompt will be replaced with the tokens
        corresponding to this text, defaults to None

    :return: (torch.LongTensor, str) Prompt tokens and Target text
    """
    source_key = "text" if "text" in json else "src"
    assert source_key in json, "json must contain a 'text' or 'src' field"

    target_key = "tgt" if "tgt" in json else None

    # NOTE: if you pass in a delimiter, we assume the target is already included
    # in the `text/src` field, and we split on the delimiter to get the prompt
    # and target respectively. That means that if you pass in a delimiter, you
    # should not pass in a `tgt` field since it will be ignored.
    if delimiter is not None and target_key is not None:
        raise AssertionError("You passed in a delimiter and a target key, but the target key will be ignored.")

    if delimiter is None:
        prompt = json[source_key]
        target = json[target_key] if target_key is not None else None
    else:
        prompt, target = json[source_key].rsplit(delimiter, 1)

    prompt = prompt_generator.get_next_prompt(prompt)
    prompt_tokens = encode_fn(prompt.rstrip(" "))
    if target is not None:
        target = target.rstrip()

    prompt_eos_tokens = encode_fn(prompt_eos_text)
    if max_prompt_len > 0:
        max_prompt_len = min(max_prompt_len - len(prompt_eos_tokens), max_seq_len - len(prompt_eos_tokens))

        prompt_tokens, _ = data_utils.truncate_source(prompt_tokens, max_prompt_len, truncation, len(prompt_eos_tokens))

    if prompt_end_text is not None and prompt_end_text != "":
        force_prompt_end_tokens = encode_fn(prompt_end_text)

        # replace last `len(force_prompt_end_tokens)` tokens with
        # `force_prompt_end_tokens` if necessary
        if prompt_tokens[-len(force_prompt_end_tokens):] != force_prompt_end_tokens:
            prompt_tokens = prompt_tokens[:-len(force_prompt_end_tokens)] + force_prompt_end_tokens

    prompt_tokens += prompt_eos_tokens
    return torch.LongTensor(prompt_tokens), target


def update_generation_config(cfg: MetaseqConfig):
    cfg.generation.sampling_topp = (cfg.generation.sampling_topp if cfg.generation.sampling_topp > 0.0 else -1.0)
    cfg.generation.sampling = cfg.generation.sampling_topp > 0.0

    assert cfg.generation.temperature >= 0.0, "temperature must be positive"
    if cfg.generation.temperature == 0.0:
        cfg.generation.sampling = False
        cfg.generation.temperature = 1.0
        cfg.generation.sampling_topp = -1


def convert_generated_tokens_to_text(
    generator_interface,
    generator,
    indexs,
    src_lengths,
    all_tokens,
    all_scores,
    all_distributions,
    all_logits,
    best_n,
    num_logprobs,
    num_logits,
    target_text,
    input_items,
    echo_prompt=False,
    copy_input_to_output=False,
    input_keys_to_copy=None,
    output_tokens_and_offsets=False,
    **kwargs,
):
    results = []
    tokens_generated = 0
    batch_size = all_tokens.size(0)
    for batch_idx in range(batch_size):
        instance_result = defaultdict(list)
        if copy_input_to_output and len(input_items) > 0:
            for key in input_keys_to_copy.split(","):
                instance_result[key] = input_items[batch_idx].get(key, None)
        instance_result["instance_idx"] = indexs[batch_idx].item()

        if len(target_text) > 0:
            instance_result["target_text"] = target_text[batch_idx]
            if output_tokens_and_offsets:
                instance_result["target_tokens"] = [
                    generator_interface.bpe.bpe.decode([t]) for t in generator_interface.encode_fn(target_text[batch_idx])
                ]

        for beam_idx in range(min(generator.beam_size, best_n)):
            # first beam is always the highest scoring
            tokens = all_tokens[batch_idx, beam_idx].tolist()
            scores = all_scores[batch_idx, beam_idx].tolist()
            distributions = all_distributions[batch_idx, beam_idx] if num_logprobs > 0 else None
            logits = all_logits[batch_idx, beam_idx] if num_logits > 0 else None

            src_length = src_lengths[batch_idx].item()
            prompt = tokens[1:src_length][:generator.max_len_b]
            if not echo_prompt:
                tokens = tokens[src_length:][:generator.max_len_b]
                scores = scores[src_length:][:generator.max_len_b]
                if num_logprobs > 0:
                    distributions = distributions[src_length:][:generator.max_len_b]
                if num_logits > 0:
                    logits = logits[src_length:][:generator.max_len_b]
                tokens_generated += len(tokens)
            else:
                tokens_generated += len(tokens) - src_length

            tokens, scores, distributions, logits = generator_interface._filter_special(
                generator_interface._pad_token_ind, generator_interface._special_token_inds, tokens, scores, distributions,
                logits
            )

            # cut off 'eos' tokens at the start
            tokens_no_eos = tokens[1:] if echo_prompt else tokens
            scores_with_eos = [None] + scores[1:] if echo_prompt else scores
            # turn it into a string
            prompt = generator_interface.bpe.bpe.decode(prompt)
            generated_text = generator_interface.bpe.bpe.decode(tokens_no_eos)
            # re-encode it so we get offsets
            token_offsets = [s for s, e in generator_interface.bpe.bpe.encode(generated_text).offsets]

            if "prompt_text" not in instance_result:
                instance_result["prompt_text"] = prompt

            beam_result = {"generated_text": generated_text}

            decoded_tokens = [generator_interface.bpe.bpe.decode([t]) for t in tokens]
            instance_length = len(decoded_tokens)
            assert instance_length == len(scores_with_eos)
            # TODO: len(generator_interface.bpe.bpe.encode(generator_interface.bpe.bpe.decode([50118, 50118]))) != len([50118, 50118])
            # assert instance_length == len(result["text_offset"])
            if output_tokens_and_offsets:
                beam_result.update(
                    {
                        "tokens": decoded_tokens,
                        # text offset is useful for cutting off prompts or prefixes
                        # or evaluating PPL on just a subset of tokens
                        "text_offset": token_offsets,
                        "token_scores": scores_with_eos,
                    }
                )

            if num_logprobs > 0:
                # final result is a List[Dict[str, float]]
                # where each item in the list corresponds to a token in the
                # sequence, and the dict provides the probabilities of the
                # top-k tokens at that timestep.
                out_logprobs = []
                all_top_scores, all_top_tokens = distributions.topk(k=num_logprobs, dim=-1)
                for top_scores, top_tokens in zip(all_top_scores, all_top_tokens):
                    logprob_item = {
                        generator_interface.bpe.bpe.decode([t.item()]): {
                            'token_id': t.item(),
                            'logprob_score': s.item()
                        }
                        for t, s in zip(top_tokens, top_scores)
                    }
                    out_logprobs.append(logprob_item)

                if echo_prompt:
                    # use null instead of giving bunk probs for EOS token
                    beam_result["top_logprobs"] = [None] + out_logprobs[1:]
                else:
                    beam_result["top_logprobs"] = out_logprobs

                # Seq generation may finish after reaching max_generation_length or after generating an <EOS>
                # token. We should account for both cases in the assertion below.
                assert instance_length in [len(beam_result["top_logprobs"]) - 1, generator.max_generation_length]

            if num_logits > 0:
                assert all_logits is not None
                # similar to top_logprobs
                out_logits = []
                all_top_logits, all_top_tokens = logits.topk(k=num_logits, dim=-1)
                for top_logits, top_tokens in zip(all_top_logits, all_top_tokens):
                    logit_item = {
                        generator_interface.bpe.bpe.decode([t.item()]): {
                            'token_id': t.item(),
                            'logit': l.item()
                        }
                        for t, l in zip(top_tokens, top_logits)
                    }
                    out_logits.append(logit_item)
                if echo_prompt:
                    # use null instead of giving zero logits for EOS token at the start
                    beam_result["top_logits"] = [None] + out_logits[1:]
                else:
                    beam_result["top_logits"] = out_logits

                # Seq generation may finish after reaching max_generation_length or after generating an <EOS>
                # token. We should account for both cases in the assertion below.
                assert instance_length in [len(beam_result["top_logits"]) - 1, generator.max_generation_length]

            instance_result["beam_results"].append(beam_result)
        results.append(instance_result)
    return results, tokens_generated


def write_inference_results_to_file(output_file: str, **kwargs):
    global TOTAL_TOKENS_GENERATED

    try:
        if distributed_utils.get_model_parallel_rank() == 0:
            batch_result, tokens_generated = convert_generated_tokens_to_text(**kwargs)
            TOTAL_TOKENS_GENERATED += (tokens_generated * distributed_utils.get_data_parallel_world_size())

            progress_bar = kwargs.pop("progress_bar")
            translate_time = kwargs.pop("translate_time")
            if distributed_utils.get_global_rank() == 0:
                progress_bar.set_postfix(
                    {
                        "Tokens Per Second":
                        f"{tokens_generated * distributed_utils.get_data_parallel_world_size() / translate_time:.2f}"
                    }
                )

            with open(output_file, "a", encoding="utf-8") as f:
                for result in batch_result:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

    except Exception as e:
        import traceback
        logger.error(f"Error in writing inference results to file.")
        logger.error(f"stack trace: {traceback.format_exc()}")


def generate(cfg: MetaseqConfig, args: argparse.Namespace):
    global TOTAL_TOKENS_GENERATED
    generator_interface = GeneratorInterface(cfg)
    generator_interface.load_model()
    logger.info(f"loaded model {cfg.distributed_training.distributed_rank}")

    logger.info(f"Resolving generator settings")
    update_generation_config(cfg)

    MAX_SEQ_LEN = utils.resolve_max_positions(
        generator_interface.task.max_positions(),
        *[model.max_positions() for model in generator_interface.models],
    )
    logger.info(f"Max sequence length for generation is set to: {MAX_SEQ_LEN}")

    logger.info("Praparing batches for generation")
    dataset, length, tgt_text, input_items = [], [], [], []
    if os.path.isdir(cfg.generation.input):
        test_files = glob(f"{cfg.generation.input}/**/*.jsonl", recursive=True)
    else:
        test_files = [cfg.generation.input]

    logger.info(f"Found the following files for generation: {test_files}")

    data_glob_path = None
    if args.few_shot_samples_data_path:
        data_glob_path = f'{args.few_shot_samples_data_path}/**/*.jsonl'

    prompt_generator = PromptGenerator(
        prompt_template_s=args.prompt_template,
        few_shot_data_glob_path=data_glob_path,
        n_fewshot_samples=args.n_few_shot,
        rng_seed=args.seed,
    )

    for file in test_files:
        json_dataset = JsonlDataset(
            path=file,
            tokenizer=partial(
                _tokenize_one_json,
                encode_fn=generator_interface.encode_fn,
                prompt_generator=prompt_generator,
                prompt_eos_text=args.prompt_eos_text,
                delimiter=args.prompt_delimiter,
                max_seq_len=MAX_SEQ_LEN,
                max_prompt_len=args.max_prompt_len,
                truncation=args.truncation,
                prompt_end_text=args.prompt_end_text,
            ),
            epoch=1,
            data_subshard_count=1,
            output_raw_items=args.copy_input_to_output
        )
        for data in json_dataset:
            if args.copy_input_to_output:
                token_data, input_data = data
                input_items.append(input_data)
            else:
                token_data = data
            ids = token_data[0]
            target_text = token_data[1]
            dataset.append(ids)
            length.append(len(ids))
            tgt_text.append(target_text)

    batches = generator_interface.task.get_batch_iterator(
        dataset=generator_interface.task.build_dataset_for_inference(dataset, length),
        max_tokens=None,
        max_sentences=cfg.dataset.batch_size,
        max_positions=None,
        ignore_invalid_inputs=False,
        skip_remainder_batch=False,
        seed=cfg.common.seed,
        data_buffer_size=cfg.generation.buffer_size,
        num_shards=distributed_utils.get_data_parallel_world_size(),
        shard_id=distributed_utils.get_data_parallel_rank(),
        num_workers=4,
    ).next_epoch_itr(shuffle=False)

    logger.info(f"Preparing generator with settings {cfg.generation}")
    args.stop_tokens = (
        [generator_interface.encode_fn(token)[0]
         for token in args.stop_tokens.split(",")] if args.stop_tokens is not None else None
    )
    generator = generator_interface.task.build_generator(
        generator_interface.models,
        cfg.generation,
        extra_gen_cls_kwargs={
            "stop": args.stop_tokens,
            "need_logprobs": args.top_k_logprobs > 0,
            "need_logits": args.top_k_logits > 0,
        },
    )
    min_len = generator.min_len
    max_len_b = generator.max_len_b

    start_time = time.time()
    total_generation_time = 1e-10

    logger.info(f'Results path: {cfg.common_eval.results_path}')

    if args.copy_input_to_output and args.input_keys_to_copy is None:
        raise argparse.ArgumentError(None, "'input_keys_to_copy' must be specified when 'copy_input_to_output' is True.")

    if distributed_utils.get_global_rank() == 0:
        progress_bar = tqdm(batches, desc="Generating")
    else:
        progress_bar = batches

    async_processor = ThreadPoolExecutor(max_workers=1)
    for batch in progress_bar:
        if batch == {}: continue
        src_lengths = batch["net_input"]["src_lengths"]

        # generator.max_generation_length is the max generation size WITHOUT the src_length.
        # E.g.: if --max_len_b 128, this would be also 128.
        generator.max_generation_length = min(MAX_SEQ_LEN, max_len_b)

        # size of the largest src item in this batch
        max_seq_len = src_lengths.max().item()

        # generator.max_len_b is the max generation size WITH the src_length.
        # E.g.: if --max_len_b 128, this would be (128 + <size of the largest src item in this batch>).
        generator.max_len_b = min(MAX_SEQ_LEN, max_len_b + max_seq_len)
        # generator.min_len is the min generation size WITH the src_length.
        # E.g.: if --min_len 1, this would be  (1 + <size of the largest src item in this batch>).
        generator.min_len = min(MAX_SEQ_LEN, min_len + max_seq_len)

        if not args.use_cpu:
            batch = utils.move_to_cuda(batch)

        translate_start_time = time.time()
        output_generation = generator_interface.task.inference_step(generator, generator_interface.models, batch)
        translate_time = time.time() - translate_start_time
        total_generation_time += translate_time

        all_tokens = output_generation["tokens"].cpu()
        all_scores = output_generation["scores"].cpu()
        if args.top_k_logprobs > 0:
            all_distributions = output_generation["distributions"].cpu(
            )  # (bsz, beam_size, seq_len + prompt_size, self.vocab_size)
        else:
            all_distributions = None

        if args.top_k_logits > 0:
            all_logits = output_generation["logits"].cpu()  # (bsz, beam_size, seq_len + prompt_size, self.vocab_size)
        else:
            all_logits = None

        batch_itemgetter = operator.itemgetter(*batch["id"].tolist())
        batch_target_text = list(batch_itemgetter(tgt_text)) if tgt_text[0] is not None else []
        batch_input_items = list(batch_itemgetter(input_items)) if args.copy_input_to_output else []

        # Async write to file
        async_processor.submit(
            write_inference_results_to_file,
            output_file=os.path.join(
                str(cfg.common_eval.results_path),
                f"worker_prediction_results_rank{distributed_utils.get_data_parallel_rank()}.jsonl"
            ),
            generator_interface=generator_interface,
            generator=generator,
            indexs=batch["id"],
            src_lengths=src_lengths,
            all_tokens=all_tokens,
            all_scores=all_scores,
            all_distributions=all_distributions,
            all_logits=all_logits,
            best_n=args.best_n,
            num_logprobs=args.top_k_logprobs,
            num_logits=args.top_k_logits,
            target_text=batch_target_text,
            input_items=batch_input_items,
            copy_input_to_output=args.copy_input_to_output,
            input_keys_to_copy=args.input_keys_to_copy,
            output_tokens_and_offsets=args.output_tokens_and_offsets,
            echo_prompt=args.echo_prompt,
            progress_bar=progress_bar,
            translate_time=translate_time,
        )

    # All processes wait for the async writes to finish at the end
    async_processor.shutdown()
    distributed_utils.global_barrier()

    if distributed_utils.get_global_rank() == 0:
        progress_bar.close()

        if args.merge_preds_on_all_ranks:
            unified_output_prediction_file = os.path.join(str(cfg.common_eval.results_path), "all_prediction_results.jsonl")
            # TODO this means that files are not sorted! In previous
            # implementation the predictions were sorted by `instance_idx`.
            # Confirm if this is really necessary (it isn't as far as I've seen)

            result_file_max_chunk_bytes_size = None
            if args.result_file_max_chunk_mb_size is not None:
                result_file_max_chunk_bytes_size = args.result_file_max_chunk_mb_size * 1024 * 1024

            current_unified_chunk_idx = 0

            def get_unified_chunk_for_idx(idx):
                # if we didn't set a max_size then return filename without
                # "chunk" part
                file_name = unified_output_prediction_file
                if result_file_max_chunk_bytes_size is not None:
                    file_name = unified_output_prediction_file.replace(".jsonl", f"_chunk{idx}.jsonl")

                return open(file_name, "w+", encoding="utf-8")

            # NOTE: we need to be careful to close this later on
            current_unified_file = get_unified_chunk_for_idx(current_unified_chunk_idx)

            workers_output_predictions_files = glob(
                os.path.join(str(cfg.common_eval.results_path), "worker_prediction_results*")
            )

            line_iterator = data_utils.multiple_file_line_generator(workers_output_predictions_files)
            for predicted_line in line_iterator:
                current_unified_file.write(predicted_line)

                # if we specified a maximum chunk size and the bytes in the
                # current chunk are already more than our max_size then close
                # current chunk and open next one
                if (
                    result_file_max_chunk_bytes_size is not None
                    and current_unified_file.tell() > result_file_max_chunk_bytes_size
                ):
                    # close current file and open next chunk
                    current_unified_file.close()
                    current_unified_chunk_idx += 1
                    current_unified_file = get_unified_chunk_for_idx(current_unified_chunk_idx)

            # close last "unified file"
            current_unified_file.close()

            logger.info(f"Written generated results to {unified_output_prediction_file}")

        logger.info(
            "Total time: {:.3f} seconds; generation time: {:.3f} seconds; avg tokens/second: {:.2f} ".format(
                time.time() - start_time,
                total_generation_time,
                TOTAL_TOKENS_GENERATED / total_generation_time,
            )
        )

        if args.metrics_list is not None or args.dataset_configuration_name is not None:
            if isinstance(args.metrics_list, str):
                args.metrics_list = args.metrics_list.split(",")
            if isinstance(args.evaluation_libraries, str):
                args.evaluation_libraries = args.evaluation_libraries.split(",")

            prediction_files_pattern = os.path.join(str(cfg.common_eval.results_path), "worker_prediction_results*")

            output_evaluation_file_path = os.path.join(str(cfg.common_eval.results_path), "evaluation_results.json")
            output_evaluation_individual_file_path = os.path.join(
                str(cfg.common_eval.results_path), "evaluation_individual_results.jsonl"
            )
            output_evaluation_exceptions_file_path = os.path.join(
                str(cfg.common_eval.results_path), "evaluation_exceptions.jsonl"
            )

            evaluate_inference_files(
                inference_file_glob_pattern=prediction_files_pattern,
                evaluation_output_file_path=output_evaluation_file_path,
                individual_results_output_file_path=output_evaluation_individual_file_path,
                exceptions_ouput_file_path=output_evaluation_exceptions_file_path,
                libraries=args.evaluation_libraries,
                metrics=args.metrics_list,
                dataset_configuration_name=args.dataset_configuration_name,
                model_configuration_name=args.model_configuration_name,
                output_metrics_for_all=args.output_metrics_for_all,
            )

        if args.save_generation_info:
            generation_info = {
                "checkpoint_name": (
                    # .../1.3b-resharded-inference-1x1/reshard.pt → 1.3b-resharded-inference-1x1
                    pathlib.Path(str(cfg.common_eval.path)).parent.name
                ),
                # .../hellaswag/valid → hellaswag
                "dataset_name":
                pathlib.Path(cfg.generation.input).parent.name,
                "num_parameters":
                sum(param.numel()
                    for param in generator.model.parameters()) * distributed_utils.get_model_parallel_world_size(),
                "generator": {
                    "vocab_size": generator.vocab_size,
                },
                "parsed_args":
                args.__dict__,
                "raw_args":
                sys.argv,
            }

            output_generation_info_file = os.path.join(str(cfg.common_eval.results_path), "generation_info.json")
            with open(output_generation_info_file, "w") as f:
                json.dump(generation_info, f)


def extra_args(parser):
    parser.add_argument("--model-dir", help="Trained checkpoint directory")
    parser.add_argument("--use-cpu", action="store_true", help="Use CPU instead for inference")

    # Prompt Parameters
    parser.add_argument(
        "--prompt-delimiter",
        type=str,
        default=None,
        help=(
            "Prompt delimiter for LM datasets. If the delimiter is specified then we'll expect a "
            "sample text to contain both the source and target texts, separated by the delimiter.",
        )
    )

    parser.add_argument("--prompt-template", type=str, default=None, help=("A Jinja2 template passed to PromptGenerator", ))
    parser.add_argument(
        "--n-few-shot",
        type=int,
        default=0,
        help="Number of examples to use for few-shot generation",
    )
    parser.add_argument(
        "--few-shot-samples-data-path",
        type=str,
        default=None,
        help="Path to a folder which contains a jsonl file from which the samples for few-shot generation are taken",
    )
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=0,
        help="Maximum number of tokens to use for the prompt. If 0, then the entire prompt will be used.",
    )
    parser.add_argument(
        "--prompt-eos-text",
        type=str,
        default="",
        help="This will be appended to the end of prompt text before generation.",
    )
    parser.add_argument(
        "--truncation",
        type=str,
        default="right",
        help="Truncation strategy for the prompt. Can be 'left', 'right', or 'none'.",
    )
    parser.add_argument(
        "--prompt-end-text",
        type=str,
        default=None,
        help=(
            "If provided, each prompt will be forced to end with this text, independently of whether the prompt was truncated or not. "
            "For example, this might be helpful if you want to ensure all your prompts end with `TLDR`. The provided tokens will overwrite "
            "the last len(prompt_end_text) tokens of the prompt."
        ),
    )

    # Generation Parameters
    parser.add_argument("--echo-prompt", action="store_true", help="Echo prompt in output")
    parser.add_argument("--stop-tokens", type=str, default=None, help="a list of terminating tokens")
    parser.add_argument(
        "--top-k-logprobs",
        type=int,
        default=0,
        help="Return this cutoff of the probability distribution. Can not be used together with 'top-k-logits' argument.",
    )
    parser.add_argument(
        "--top-k-logits",
        type=int,
        default=0,
        help="Return the top K logits in the inference output file. Can not be used together with 'top-k-logprobs' argument.",
    )
    parser.add_argument("--best-n", type=int, default=1, help="return this cutoff of the beam search")
    parser.add_argument("--merge-preds-on-all-ranks", action="store_true", help="merge prediction results on all ranks")
    parser.add_argument("--copy-input-to-output", action="store_true", help="copy input to output")
    parser.add_argument("--input-keys-to-copy", type=str, default=None, help="comma separated list of input keys to copy")
    parser.add_argument("--output-tokens-and-offsets", action="store_true", help="output tokens and offsets")

    # Metrics
    parser.add_argument(
        "--evaluation-libraries",
        type=str,
        default="parlai",
        help=
        f"name of the library that should be used to compute the evaluation metrics. Possible options: {GenerationMetrics.metric_libraries}"
    )
    parser.add_argument(
        "--metrics-list",
        type=str,
        default=None,
        help="comma separated list of metrics to calculate",
    )
    parser.add_argument(
        "--dataset-configuration-name",
        type=str,
        default=None,
        help="If provided then metric prameters will be obtained from this dataset configuration",
    )
    parser.add_argument(
        "--model-configuration-name",
        type=str,
        default=None,
        help=
        "If provided then this model configuration will be obtained from the dataset configuration and used to compute metrics",
    )
    parser.add_argument(
        "--result-file-max-chunk-mb-size",
        type=float,
        default=None,
        help=
        "If provided, the inference results will be separated accross many files, each being chunked with the specified size",
    )
    parser.add_argument(
        "--output-metrics-for-all",
        action="store_true",
        help="output all metrics for each instance",
    )
    parser.add_argument("--pretty-metrics", action="store_true", help="pretty print metrics")
    parser.add_argument(
        "--save-generation-info",
        action="store_true",
        help="if provided then a file with information on the parameters used for generation will also be saved"
    )

    return parser


def cli_main():
    """
    Generation using trained model.
    """
    parser = options.get_generation_parser()
    parser = extra_args(parser)

    # dumb defaults overriding
    parser.set_defaults(
        lr_scheduler=None, criterion=None, task="language_modeling", bpe="hf_byte_bpe", arch="transformer_lm_megatron"
    )
    args = options.parse_args_and_arch(parser)

    # set args
    args.bpe_vocab = args.vocab_filename
    args.bpe_merges = args.merges_filename
    args.path = args.model_dir

    # Output log for RANK 0 process. All ranks perform inference in data parallel so we record
    # the log only for one process.
    if os.environ.get("RANK", "0") == "0":
        # Clean up results directory
        shutil.rmtree(args.results_path, ignore_errors=True)

        if not args.log_file:
            args.log_file = "evaluation.log"
        args.log_file = os.path.join(args.results_path, args.log_file)
        if os.path.dirname(args.log_file) != "":
            os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        handler = logging.FileHandler(filename=args.log_file)
        logger.addHandler(handler)

    cfg = convert_namespace_to_omegaconf(args)
    if os.environ.get("RANK", "0") == "0":
        os.makedirs(os.path.join(args.results_path, "config"), exist_ok=True)
        OmegaConf.save(
            config=flatten_config(cfg),
            f=os.path.join(args.results_path, "config", "config.yml"),
        )

    os.environ["NCCL_DEBUG"] = "WARN"
    distributed_utils.call_main(cfg, generate, args=args)


if __name__ == "__main__":
    cli_main()
