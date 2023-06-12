from collections import defaultdict
from typing import Dict, List, Union

from metaseq.generation_metrics.wrappers.coco import measure_scores
from metaseq.generation_metrics.wrappers.coco.measure_scores import CustomCOCOEvalCap, ResultsForPrompt
from metaseq.logging import get_logger

logger = get_logger(__name__)


class CocoMetrics:
    """
    A note on these metrics is that they'll automatically perform *grouping* of
    the items to be evaluted by their ``prompt``. The *generated text* is then
    compared against all ``target texts`` related with the prompt at once and
    the average is returned.

    Existing metrics:

    - bleu
    - meteor
    - rouge-L / rouge
    - cider
    - spice
    - nist
    """

    allowed_metrics = CustomCOCOEvalCap.allowed_metrics.union({"nist"})

    def __init__(self, metrics: List[str] = []) -> None:

        # if empty list then assume we want to run all metrics
        if len(metrics) == 0:
            metrics = list(CocoMetrics.allowed_metrics)

        # ignore metrics that we don't know about. Not an error since these
        # other metrics might be used by other "evaluators" (e.g. ParlAI)
        final_metrics = set(metrics).intersection(CocoMetrics.allowed_metrics)

        logger.info("Registering COCO evaluator with the following list of metrics: %s", final_metrics)
        self.metrics_list = final_metrics

        self.prompt_to_results: Dict[
            str, ResultsForPrompt] = defaultdict(lambda: ResultsForPrompt(
                prompt="",
                generated_text="",
                target_texts=[],
            ))

    def __call__(self, prompt: str, prediction: str, label: Union[List[str], str]) -> Dict:
        prompt_result = self.prompt_to_results[prompt]
        prompt_result["prompt"] = prompt

        # only keep the first generated_text
        if prompt_result["generated_text"] == "":
            prompt_result["generated_text"] = prediction

        if type(label) == list:
            prompt_result["target_texts"].extend(label)
        else:
            prompt_result["target_texts"].append(label)  # type: ignore

        # TODO is this ok? Returning metrics just for every sample would be
        # really slow. Does it even make sense considering we're assured to only
        # have access to a single target?
        return {}

    def accumulate(self) -> Dict:
        return measure_scores.evaluate(
            list(self.prompt_to_results.values()),
            metrics=self.metrics_list,
        )
