import logging
from typing import Dict, List, Union

import evaluate

from metaseq.logging import get_logger

logger = get_logger(__name__)


class HFEvaluateMetrics():
    """
    Existing Metrics:

    .. code-block:: python

        [
            'precision', 'code_eval', 'roc_auc', 'cuad', 'xnli', 'rouge', 'pearsonr', 'mse', 'super_glue',
            'comet', 'cer', 'sacrebleu', 'mahalanobis', 'wer', 'competition_math', 'f1', 'recall', 'coval',
            'mauve', 'xtreme_s', 'bleurt', 'ter', 'accuracy', 'exact_match', 'indic_glue', 'spearmanr', 'mae',
            'squad', 'chrf', 'glue', 'perplexity', 'mean_iou', 'squad_v2', 'meteor', 'bleu', 'wiki_split', 'sari',
            'frugalscore', 'google_bleu', 'bertscore', 'matthews_correlation', 'seqeval','trec_eval', 'rl_reliability',
            'poseval', 'brier_score', 'mase', 'mape', 'smape', 'nist_mt', 'character', 'charcut_mt', 'mcnemar',
            'exact_match', 'wilcoxon', 'word_length', 'word_count', 'text_duplicates', 'perplexity', 'label_distribution',
            'toxicity', 'regard', 'honest'
        ]
    """

    def __init__(self, metrics: List[str], **kwargs) -> None:
        self.metrics = metrics
        self.infer_metrics(**kwargs)

    def infer_metrics(self, **kwargs) -> None:
        self.metric_cls = {}
        for metric in self.metrics:
            try:
                self.metric_cls[metric] = evaluate.load(metric, **kwargs)
            except Exception:
                logger.warning(f"HF Evaluate: Metric {metric} is not supported.")

    def __call__(self, _prompt: str, prediction: str, label: Union[List[str], str]) -> Dict:
        response = {}
        for metric, metric_cls in self.metric_cls.items():
            kwargs = {}
            if metric == "perplexity":
                # TODO: Update this to not use a hardcoded default model_id
                kwargs = {"model_id": "gpt2"}

            response.update(metric_cls.compute(predictions=[prediction], references=[label], **kwargs))
        return response

    def accumulate(self) -> Dict:
        logging.warning("Accumulate is not supported for HF Evaluate Metrics yet.")
        return {}
