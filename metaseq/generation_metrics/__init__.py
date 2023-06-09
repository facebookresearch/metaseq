import importlib
import os

from .coco_metrics import CocoMetrics
from .hf_metrics import HFEvaluateMetrics
from .metrics import GenerationMetrics
from .parlai_metrics import ParlAiMetrics

__all__ = ["GenerationMetrics", "HFEvaluateMetrics", "ParlAiMetrics", "CocoMetrics"]

# automatically import any Python files/directories in the tasks/ directory
tasks_dir = os.path.join(os.path.dirname(__file__), "tasks")

if os.path.exists(tasks_dir):
    for item in os.listdir(tasks_dir):
        path = os.path.join(tasks_dir, item)
        if (not item.startswith("_") and not item.startswith(".") and (item.endswith(".py") or os.path.isdir(path))):
            task_name = item[:item.find(".py")] if item.endswith(".py") else item
            module = importlib.import_module("metaseq.generation_metrics.tasks." + task_name)
