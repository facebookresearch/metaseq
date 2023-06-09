import os
from glob import glob
import json
from typing import Any, Dict, List, Optional, Union
import fire
from datetime import datetime

from metaseq.data import data_utils
from metaseq.data.datasets.dataset_configurations import DATASET_CONFIGURATIONS
from metaseq.data.datasets.types import MetaseqInferenceOutputItem

from metaseq.generation_metrics.coco_metrics import CocoMetrics
from metaseq.generation_metrics.grindstone_metrics import GrindstoneMetrics
from metaseq.generation_metrics.hf_metrics import HFEvaluateMetrics
from metaseq.generation_metrics.parlai_metrics import ParlAiMetrics
from metaseq.logging import get_logger
from metaseq.scripts import script_utils

repo_root = script_utils.get_repo_root_path()
logger = get_logger(__name__)


class GenerationMetrics():
    metric_libraries = {"all", "parlai", "hf", "coco", "grindstone"}

    def __init__(self, metrics: List[str], libraries: List[str] = ["parlai"], **kwargs) -> None:
        self.metrics = [m.strip() for m in metrics]
        self.libraries = [l.lower().strip() for l in libraries]
        self.infer_metrics(**kwargs)

    def infer_metrics(self, **kwargs) -> None:
        self.metric_cls = {}

        for library in self.libraries:
            assert library in self.metric_libraries, f"library {self.libraries} not supported"

        if "all" in self.libraries or "parlai" in self.libraries:
            # parlai expects metrics list as a string
            metrics = ",".join(self.metrics)
            self.metric_cls["parlai"] = ParlAiMetrics(metrics_list=metrics, **kwargs)

        if "all" in self.libraries or "hf" in self.libraries:
            self.metric_cls["hf"] = HFEvaluateMetrics(metrics=self.metrics, **kwargs)

        if "all" in self.libraries or "coco" in self.libraries:
            self.metric_cls["coco"] = CocoMetrics(metrics=self.metrics, **kwargs)

        if "all" in self.libraries or "grindstone" in self.libraries:
            self.metric_cls["grindstone"] = GrindstoneMetrics(metrics=self.metrics, **kwargs)

    def __call__(self, prompt: str, prediction: str, label: Union[List[str], str]) -> Dict[str, float]:
        response = {}
        for metric_lib, metric_cls in self.metric_cls.items():
            response[metric_lib] = metric_cls(prompt, prediction, label)
        return response

    def accumulate(self) -> Dict[str, float]:
        # TODO: "accumulate" is not supported by HFEvaluateMetrics
        assert "hf" not in self.metric_cls, "'accumulate' is not supported by HFEvaluateMetrics"

        response = {}
        for metric_lib, metric_cls in self.metric_cls.items():
            response[metric_lib] = metric_cls.accumulate()

        return response


def evaluate_inference_files(
    inference_file_glob_pattern: str,
    evaluation_output_file_path: str,
    individual_results_output_file_path: str,
    exceptions_ouput_file_path: str,
    libraries: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    dataset_configuration_name: Optional[str] = None,
    model_configuration_name: Optional[str] = None,
    output_metrics_for_all: bool = False,
):
    """
    Evaluate inference files using the provided metrics and libraries, or using
    the provided dataset and model configuration.

    :param str inference_file_glob_pattern: Glob pattern that will be used to
        find the inference files to evaluate.
    :param str evaluation_output_file: Path to the file where the evaluation
        results will be written.
    :param str exceptions_ouput_file: Path to the file where any exceptions that
        happen during evaluation will be recorded.
    :param Optional[List[str]] libraries: List of libraries to use during
        evaluation, defaults to None
    :param Optional[List[str]] metrics: List of metrics to use during
        evaluation, defaults to None
    :param Optional[str] dataset_configuration_name: The dataset configuratio
        name to use during evaluation. If provided AND this configuration has
        `libraries` and `metrics` then this will overwrite the `libraries` and
        `metrics` args with the values of the configuration, defaults to None
    :param Optional[str] model_configuration_name: Name of the model
        configuration to get from the dataset configuration, defaults to None
    :param bool output_metrics_for_all: If True then the result file will
        contain a line for the evaluation result of each inference, defaults to
        False
    """

    if model_configuration_name is not None:
        assert dataset_configuration_name is not None, "Expected dataset configuration name to be provided when model configuration name is provided"

    # initialize evaluator
    if dataset_configuration_name is not None:
        logger.info(f"Using dataset configuration: {dataset_configuration_name}")

        dataset_config = DATASET_CONFIGURATIONS[dataset_configuration_name]
        if dataset_config.common.metric_names is not None:
            logger.info(f"Overriding metrics with dataset configuration: {dataset_config.common.metric_names}")
            metrics = dataset_config.common.metric_names

        if dataset_config.common.metric_libraries is not None:
            logger.info(
                f"Overriding evaluation libraries with dataset configuration: {dataset_config.common.metric_libraries}"
            )
            libraries = dataset_config.common.metric_libraries

    assert libraries is not None, "Expected evaluation libraries to be provided when running evaluation"
    assert metrics is not None, "Expected evaluation metrics to be provided when running evaluation"

    # load hook (or default to identity)
    model_domain_to_original_domain_transformer = lambda x: x
    if dataset_configuration_name is not None and model_configuration_name is not None:
        model_domain_to_original_domain_transformer = (
            DATASET_CONFIGURATIONS[dataset_configuration_name].model_config.model_hooks.
            convert_model_type_output_to_original_domain[model_configuration_name]
        )

    # create evaluator
    evaluator = GenerationMetrics(metrics=metrics, libraries=libraries)

    # load inference files
    inference_files = glob(inference_file_glob_pattern, recursive=True)
    assert len(
        inference_files
    ) > 0, f"Found no files to evaluate on that match the provided pattern: {inference_file_glob_pattern}"
    line_iterator = data_utils.multiple_file_line_generator(inference_files)

    # evaluate
    with open(evaluation_output_file_path, "w") as f_results, \
        open(individual_results_output_file_path, "w") as f_individual_results, \
        open(exceptions_ouput_file_path, "w") as f_exceptions:
        total_num_exceptions = 0
        total_num_evaluated_successfully = 0

        for predicted_line in line_iterator:
            inference_row: MetaseqInferenceOutputItem = json.loads(predicted_line)

            row_metrics = []
            target_text = inference_row["target_text"]

            for beam_idx, beam in enumerate(inference_row["beam_results"]):

                model_output = beam["generated_text"]

                try:
                    model_output = model_domain_to_original_domain_transformer(model_output)

                    row_metrics.append(
                        evaluator(
                            prompt=inference_row["prompt_text"],
                            prediction=model_output,
                            label=target_text,
                        )
                    )

                    total_num_evaluated_successfully += 1
                except Exception as e:
                    logger.error(
                        f"Exception attempting evaluate line {line_iterator.current_line_num}\n"
                        f"Exception: {e}\n"
                        f"Raw item:\n\t{inference_row}"
                    )
                    total_num_exceptions += 1

                    f_exceptions.write(
                        json.dumps(
                            {
                                "file": line_iterator.current_file_path,
                                "line_number": line_iterator.current_line_num,
                                "error": f"{type(e).__name__} - {e}",
                                "raw_line": predicted_line,
                                "beam_idx": beam_idx,
                            }
                        ) + "\n"
                    )

            if output_metrics_for_all:
                f_individual_results.write(json.dumps(row_metrics) + "\n")

        accumulated_metrics: Dict[str, Any] = evaluator.accumulate()
        accumulated_metrics["evaluation_info"] = {
            "total_num_evaluated_successfully": total_num_evaluated_successfully,
            "total_num_exceptions": total_num_exceptions,
        }

        f_results.write(json.dumps(accumulated_metrics, indent=2))

        logger.info(f"Saved evaluation results to {evaluation_output_file_path}")
        total_rows = total_num_exceptions + total_num_evaluated_successfully
        if total_num_exceptions > 0:
            logger.info(f"Saved evaluation exceptions to {exceptions_ouput_file_path}")
            logger.info(
                f"Total number of exceptions encountered: {total_num_exceptions} ({total_num_exceptions / total_rows * 100:0.2f}% of total rows)"
            )
        logger.info(
            f"Total number of inference rows evaluated successfully: {total_num_evaluated_successfully} ({total_num_evaluated_successfully / total_rows * 100:0.2f}% of total rows)"
        )
        logger.info(f"(Accumulated Metrics)\n{json.dumps(accumulated_metrics, indent=2)}")


def cli_main(
    input_file_path: str,
    output_folder_path=repo_root / '_results',
    libraries: Union[str, List[str]] = ["parlai", "grindstone"],
    metrics: Union[str, List[str]] = ["rouge-L", "bertscore"],
    pretty: bool = False,
    output_all: bool = False,
    dataset_configuration_name: Optional[str] = None,
    model_configuration_name: Optional[str] = None,
    prompt: str = "",
    label: Optional[str] = None,
    prediction: Optional[str] = None,
):
    """
    This script can be used in two ways:

    1. Pass it an input file which is obtained from Metaseq's inference
       component, and it will evaluate how good are the model's predictions vs
       the ground truth label.
    2. Pass it a single triplet of "prompt", "label" (aka 'ground truth') and
       "prediction" and it will print out the result of evaluating that single
       triplet. This is mainly used for quick testing.

    **Example usage:**

    **Evaluating inference results directly:**

    .. code-block:: bash

        python -m metaseq.generation_metrics.metrics \\
            --pretty=True \\
            --metrics="rouge-L,bertscore" \\
            --libraries="coco,grindstone" \\
            --input-file-path="/mnt/input_data_dir/examples/evaluation/all_prediction_results.jsonl"


    **Evaluating inference results using dataset configuration:**

    .. code-block:: bash

        python -m metaseq.generation_metrics.metrics \\
            --pretty=True \\
            --dataset_configuration_name="hellaswag" \\
            --model_configuration_name="distilled" \\
            --input-file-path="/mnt/input_data_dir/examples/evaluation/all_prediction_results.jsonl"

    :param str input_file_path: This is the path to an inference file that was
        produced by Metaseq's inference component.
    :param str output_folder_path: The path to folder which will contain the results of
        evaluating the items in `input_file`.
    :param str libraries: A comma-separated list of the libraries we want to use
        to compute the evaluation. Allowed values are [``parlai``,
        ``grindstone``, ``hf``, ``all``]. If ``all`` is provided then *all*
        libraries will be used. The final result will have an entry for each of
        the libraries. Defaults to "parlai,grindstone"
    :param str metrics: A comma-separated list of the metrics that we want to
        compute. If the librar(y/ies) being used support this metric then it
        will be included among the metrics computed by said library. This means
        that if multiple libraries are chosen and a subset of them supports a
        given metric then it will be computed for all libraries in the subset,
        defaults to "bleu,rouge-L"
    :param bool pretty: If True then the final result will be pretty-printed,
        defaults to False
    :param bool output_all: If True then the evaluation score of every item will
        be returned together with the aggregate. If false, only the aggregate is
        returned, defaults to False
    :param Optional[str] dataset_configuration_name: If provided then the
        evaluation configuration will be taken from the dataset config with this
        name, defaults to None
    :param Optional[str] model_configuration_name: If provided then this model
        configuration will be obtained from the dataset configuration and used
        to compute metrics, defaults to None
    :param str prompt: To be used when you desire to do an evalution of a single
        triplet. Represents the prompt. Defaults to ""
    :param Optional[str] label: To be used when you desire to do an evalution of
        a single triplet. Represents the label/ground truth. Defaults to None
    :param Optional[str] prediction: To be used when you desire to do an
        evalution of a single triplet. Represents the prediction/generated text.
        Defaults to None
    """

    # convert metrics and libraries args to lists if they are not already
    if isinstance(metrics, str):
        metrics = metrics.split(",")

    if isinstance(libraries, str):
        libraries = libraries.split(",")

    output_folder_path, [aggregated_results_file_path, individual_results_file_path,
                         exceptions_file_path] = script_utils.create_output_folder_and_get_filepaths(
                             output_folder_path, 'metrics_evaluation',
                             [f'aggregated_results.json', f'individual_results.jsonl', f'exceptions.jsonl']
                         )

    if input_file_path is not None and output_folder_path is not None:
        evaluate_inference_files(
            inference_file_glob_pattern=input_file_path,
            evaluation_output_file_path=aggregated_results_file_path,
            individual_results_output_file_path=individual_results_file_path,
            exceptions_ouput_file_path=exceptions_file_path,
            libraries=libraries,
            metrics=metrics,
            dataset_configuration_name=dataset_configuration_name,
            model_configuration_name=model_configuration_name,
            output_metrics_for_all=output_all,
        )

    elif prediction is not None and label is not None:
        gen_metrics = GenerationMetrics(metrics=metrics, libraries=libraries)
        results = gen_metrics(prompt, prediction, label)
        if pretty:
            print(json.dumps(results, indent=2))
        else:
            print(results)

    else:
        raise ValueError("Either input_file_path and output_folder_path or prediction and label must be provided")


if __name__ == "__main__":
    fire.Fire(cli_main)
