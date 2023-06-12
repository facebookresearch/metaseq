from typing import Dict, List, Optional, Union

from parlai.core.metrics import Metric, TeacherMetrics

from metaseq.logging import get_logger

logger = get_logger(__name__)


class ParlAiMetrics(TeacherMetrics):
    """
    Existing Metrics:

    .. code-block:: python

        [
            'accuracy', 'auc', 'bleu-4', 'clen', 'clip', 'ctpb', 'ctps', 'ctrunc', 'ctrunclen',
            'exps', 'exs', 'f1', 'gen_n_toks', 'gnorm', 'gpu_mem', 'hits@1', 'hits@5',
            'interdistinct-1', 'interdistinct-2', 'intradistinct-1', 'intradictinct-2',
            'jga', 'llen', 'loss', 'lr', 'ltpb', 'ltps', 'ltrunc', 'ltrunclen', 'precision',
            'recall', 'rouge-1', 'rouge-2', 'rouge-L', 'token_acc', 'token_em', 'total_train_updates',
            'tpb', 'tps', 'ups'
        ]

    - DEFAULT_METRICS = {'bleu-4', 'accuracy', 'f1'}
    - ROUGE_METRICS = {'rouge-1', 'rouge-2', 'rouge-L'}
    - BLEU_METRICS = {'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4'}
    - DISTINCT_METRICS = {'interdistinct-1', 'interdistinct-2', 'intradistinct-1', 'intradistinct-2'}

    Alias:

    .. code-block:: python

        [
            'default', 'rouge', 'bleu', 'distinct', 'all'
        ]
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, _prompt: str, prediction: str, label: Union[List[str], str]) -> Dict:
        self.evaluate_response(observation={'text': prediction}, labels=[label] if isinstance(label, str) else label)
        return {k: v.value() for k, v in self.report_recent().items()}

    def accumulate(self) -> Dict:
        return {k: v.value() for k, v in self.report().items()}

    def add(self, key: str, value: Optional[Metric]) -> None:
        """
        Record an accumulation to a metric.
        """
        # Fixing bug with self._recent_data in parlai.core.metrics.TeacherMetrics.add
        self._data[key] = self._data.get(key) + value
        self._recent_data[key] = value
