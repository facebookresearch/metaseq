from collections import defaultdict
from typing import Callable, Dict, List, Union
from bert_score import BERTScorer
import numpy as np
from rouge_score import rouge_scorer

EvalFunctionType = Callable[[str, List[str]], Dict[str, float]]


def _build_rouge_function(rouge_types: List[str]) -> EvalFunctionType:
    """
    Returns a function that will compute the Rouge metrics for a pair of
    "prediction" and "references".

    These are computed using the same library and params that are used in the
    HELM benchamark.

    :param List[str] rouge_types: Rouge types we want to have this evaluator
        return. These come from :ref:`rouge_scorer.RougeScorer`.
    :return EvalFunctionType: evaluation function that returns a score for each
        of the :param rouge_types:.
    """
    # Use the same rouge scorers that are used in Helm
    # https://github.dev/stanford-crfm/helm/blob/a1c5e4293d6b475aca567abaadf4eae8c978bd4e/src/helm/benchmark/metrics/summarization_metrics.py#L46-L51
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    def evaluate_closure(pred: str, references: List[str]) -> Dict[str, float]:
        # https://github.dev/stanford-crfm/helm/blob/a1c5e4293d6b475aca567abaadf4eae8c978bd4e/src/helm/benchmark/metrics/summarization_metrics.py#L131-L137
        max_scores = defaultdict(lambda: -np.inf)
        for reference in references:
            score = scorer.score(prediction=pred, target=reference)

            for rouge_type in rouge_types:
                score_for_type = score[rouge_type].fmeasure
                if score_for_type > max_scores[rouge_type]:
                    max_scores[rouge_type] = score_for_type

        return max_scores

    return evaluate_closure


def _build_bert_scorer() -> EvalFunctionType:
    """
    Returns a function that will compute the BERTScore for a pair of
    "prediction" and "references".

    These are computed using the same library and params that are used in the
    HELM benchamark.

    :return EvalFunctionType: evaluation function that returns the Precision,
        Recall, and F1, from the BERTScorer.
    """
    # same params as Helm:
    # https://github.dev/stanford-crfm/helm/blob/a1c5e4293d6b475aca567abaadf4eae8c978bd4e/src/helm/benchmark/metrics/summarization_metrics.py#L68-L69
    scorer = BERTScorer(
        model_type="microsoft/deberta-large-mnli",
        lang="en",
        rescale_with_baseline=True,
    )

    def evaluate_closure(pred: str, references: List[str]) -> Dict[str, float]:
        # https://github.dev/stanford-crfm/helm/blob/a1c5e4293d6b475aca567abaadf4eae8c978bd4e/src/helm/benchmark/metrics/summarization_metrics.py#L150-L152
        precision, recall, f1score = scorer.score(cands=[pred], refs=[references])
        return {"BERTScore-P": precision[0].item(), "BERTScore-R": recall[0].item(), "BERTScore-F": f1score[0].item()}

    return evaluate_closure


def _build_accuracy_scorer() -> EvalFunctionType:

    # Scorer comes directly from HELM:
    # https://github.dev/stanford-crfm/helm/blob/80ecb204a9fed6a54a53e1f5950e9c1e145acddc/src/helm/benchmark/metrics/basic_metrics.py#L137-L138
    def exact_match(gold: str, pred: str) -> float:
        if not pred:
            return 0

        return 1 if gold.strip() == pred.strip() else 0

    def evaluate_closure(pred: str, references: List[str]) -> Dict[str, float]:
        computed_exact_match = np.mean([exact_match(ref, pred) for ref in references])

        return {"accuracy": computed_exact_match}

    return evaluate_closure


class GrindstoneMetrics:
    """
    This class presents a collection of metrics+evaluators with the goal of
    ensuring the algorithms we use here are the same as those used in popular
    benchmarks (e.g. HELM, BabelBench) so we can properly compare against them.
    """

    def __init__(self, metrics: List[str]) -> None:
        self.metrics = set(m.lower() for m in metrics)

        if "rouge-l" in self.metrics:
            # Add all Rouge-* if rouge-L is provided so we're compatible with
            # other evaluators
            self.metrics.add("rouge")

        # the sum of all results of a given metric
        self.sum_of_metrics: Dict[str, float] = defaultdict(lambda: 0.0)

        # how many examples we've seen
        self.exs = 0

        # register metric functions
        self.metric_name_to_eval_fn: Dict[str, EvalFunctionType] = {}
        if "bertscore" in self.metrics:
            self.metric_name_to_eval_fn["BERTScore"] = _build_bert_scorer()

        if "rouge" in self.metrics:
            self.metric_name_to_eval_fn["rouge"] = _build_rouge_function(["rouge1", "rouge2", "rougeL"])

        if "accuracy" in self.metrics:
            self.metric_name_to_eval_fn["accuracy"] = _build_accuracy_scorer()

    def __call__(self, _prompt: str, prediction: str, label: Union[List[str], str]) -> Dict:
        response = {}

        label_list: List[str]
        if type(label) == str:
            label_list = [label]  # type: ignore
        else:
            label_list = label  # type: ignore

        self.exs += 1

        # get metrics from each of the selected metrics classes
        for eval_fn in self.metric_name_to_eval_fn.values():
            output = eval_fn(prediction, label_list)

            for metric, score in output.items():
                # self.sum_of_metrics is a defaultdict(0.0), so score always
                # starts at 0
                self.sum_of_metrics[metric] += score

            response.update(output)

        return response

    def accumulate(self) -> Dict:
        result = {}

        for metric, accumulated_score in self.sum_of_metrics.items():
            result[metric] = accumulated_score / self.exs

        return result
