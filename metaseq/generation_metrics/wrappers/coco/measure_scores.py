# Adapted from https://github.com/tuetschek/e2e-metrics/blob/master/measure_scores.py

#!/usr/bin/env python3

from __future__ import print_function

import json
import sys
from builtins import str, zip
from collections import defaultdict
from glob import glob
from pprint import pp
from typing import List, Set, TypedDict
import fire
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from .pymteval import BLEUScore, NISTScore
from metaseq.generation_metrics import grindstone_metrics


class ResultsForPrompt(TypedDict):
    """
    Represents the result of a given prompt with its generated text and all the
    target senteces that belong to this prompt.
    """
    prompt: str
    generated_text: str
    target_texts: List[str]


class CustomCOCORouge(Rouge):
    """
    Here we're overriding COCO's default Rouge implementation since it differs
    from the implementation used in Helm and BabelBench. Specifically, the
    difference is that the algorithm uses a different tokenizer. This causes
    only small discrepancies in the final score, but there's no need to
    introduce extra noise (in the score).

    :param _type_ Rouge: _description_
    """

    def __init__(self):
        self.inner_scorer = grindstone_metrics._build_rouge_function(["rougeL"])

    def calc_score(self, candidate, refs):
        # ensure input has the correct shape
        if type(candidate) == list:
            candidate = candidate[0]

        if type(refs) == str:
            refs = [refs]

        result = self.inner_scorer(pred=candidate, references=refs)
        return result["rougeL"]


class CustomCOCOEvalCap(COCOEvalCap):
    """
    This is a reimplementation of pycocoevalcap.eval.COCOEvalCap which allows us
    to specify which metrics we want to run.
    """

    allowed_metrics = {"bleu", "meteor", "rouge-L", "rouge", "cider", "spice"}

    def __init__(self, coco, cocoRes, metrics: Set[str] = allowed_metrics):
        if "rouge" in metrics:
            # rouge is a supergroup that contains the following
            metrics.add("rouge-L")

        assert len(metrics.difference(CustomCOCOEvalCap.allowed_metrics)) == 0, (
            "The provided list of metrics to CustomCOCOEvalCap is invalid. "
            f"Allowed metrics are: {CustomCOCOEvalCap.allowed_metrics}"
        )

        super().__init__(coco, cocoRes)
        self.metrics_list = metrics

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')

        # maps the name of the metric we want to calculate to a lazy function
        # which returns the instance of the scorer so that we can avoid
        # initializing scorers that we won't end up using.
        metric_name_to_scorers = {
            "bleu": (lambda: Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            "meteor": (lambda: Meteor(), "METEOR"),
            "rouge-L": (lambda: CustomCOCORouge(), "ROUGE_L"),
            "cider": (lambda: Cider(), "CIDEr"),
            "spice": (lambda: Spice(), "SPICE")
        }

        # =================================================
        # Compute scores
        # =================================================
        for metric in self.metrics_list:
            if metric in metric_name_to_scorers:
                get_scorer, method = metric_name_to_scorers[metric]
                scorer = get_scorer()

                print('computing %s score...' % (scorer.method()))
                score, scores = scorer.compute_score(gts, res)
                if type(method) == list:
                    for sc, scs, m in zip(score, scores, method):
                        self.setEval(sc, m)
                        self.setImgToEvalImgs(scs, gts.keys(), m)
                        print("%s: %0.3f" % (m, sc))
                else:
                    self.setEval(score, method)
                    self.setImgToEvalImgs(scores, gts.keys(), method)
                    print("%s: %0.3f" % (method, score))

        self.setEvalImgs()


def create_coco_refs(data_ref):
    """Create MS-COCO human references JSON."""
    out = {'info': {}, 'licenses': [], 'images': [], 'type': 'captions', 'annotations': []}
    ref_id = 0
    for inst_id, refs in enumerate(data_ref):
        out['images'].append({'id': 'inst-%d' % inst_id})
        for ref in refs:
            out['annotations'].append({'image_id': 'inst-%d' % inst_id, 'id': ref_id, 'caption': ref})
            ref_id += 1
    return out


def create_coco_sys(data_sys):
    """Create MS-COCO system outputs JSON."""
    out = []
    for inst_id, inst in enumerate(data_sys):
        out.append({'image_id': 'inst-%d' % inst_id, 'caption': inst})
    return out


def run_pymteval(data_ref, data_sys, metrics: Set[str]):
    """Run document-level BLEU and NIST in their Python implementation (should give the
    same results as Perl)."""
    print('Running Py-MTEval metrics...', file=sys.stderr)

    bleu = None
    if "bleu" in metrics:
        bleu = BLEUScore()

    nist = None
    if "nist" in metrics:
        nist = NISTScore()

    # collect statistics
    for sents_ref, sent_sys in zip(data_ref, data_sys):
        if bleu:
            bleu.append(sent_sys, sents_ref)

        if nist:
            nist.append(sent_sys, sents_ref)

    result = {}

    if bleu:
        result["BLEU"] = bleu.score()

    if nist:
        result["NIST"] = nist.score()

    return result


def run_coco_eval(data_ref, data_sys, metrics: Set[str]):
    """Run the COCO evaluator, return the resulting evaluation object (contains both
    system- and segment-level scores."""
    # convert references and system outputs to MS-COCO format in-memory
    coco_ref = create_coco_refs(data_ref)
    coco_sys = create_coco_sys(data_sys)

    print('Running MS-COCO evaluator...', file=sys.stderr)
    coco = COCO()
    coco.dataset = coco_ref
    coco.createIndex()

    coco_res = coco.loadRes(resFile=coco_sys)

    coco_metrics = metrics.intersection(CustomCOCOEvalCap.allowed_metrics)
    coco_eval = CustomCOCOEvalCap(coco, coco_res, coco_metrics)
    coco_eval.evaluate()

    return coco_eval


def evaluate(results: List[ResultsForPrompt], metrics: Set[str]):
    """Main procedure, running the MS-COCO & MTEval evaluators on the loaded data."""

    # generated[i], and targets[i] all correspond to the i'th sample
    # in the data
    all_generated: List[str] = []
    all_targets: List[List[str]] = []

    for res in results:
        all_generated.append(res["generated_text"])
        all_targets.append(res["target_texts"])

    assert len(all_generated) == len(all_targets)

    # run the MS-COCO evaluator
    coco_eval = run_coco_eval(all_targets, all_generated, metrics)
    scores = {metric: score for metric, score in list(coco_eval.eval.items())}

    # run MT-Eval
    mteval_scores = run_pymteval(all_targets, all_generated, metrics)

    scores.update(mteval_scores)

    return scores


def get_prediction_results(all_predictions_file_glob: str) -> List[ResultsForPrompt]:
    res = defaultdict(lambda: ResultsForPrompt(
        prompt="",
        generated_text="",
        target_texts=[],
    ))

    all_prediction_files = sorted(glob(all_predictions_file_glob, recursive=True))

    for pred_file_path in all_prediction_files:
        for line in open(pred_file_path, "r"):
            data = json.loads(line)

            prompt: str = data["prompt_text"]
            target: str = data["target_text"]

            # TODO here we're taking the BEST from the beam. But we could also
            #      take all results in the beam. Any ideas?
            generated: str = data["beam_results"][0]['generated_text']

            res_for_this_prompt = res[prompt]
            res_for_this_prompt["prompt"] = prompt
            res_for_this_prompt["target_texts"].append(target)
            res_for_this_prompt["generated_text"] = generated

    return list(res.values())


def cli_main(all_predictions_file_glob: str, output_path: str = "_results/coco_evaluation_results.json"):
    """
    E2E Challenge evaluation -- MS-COCO & MTEval wrapper

    :param str all_predictions_file_glob: Glob string to be used to find all the
        predictions we want to evaluate
    :param str output_path: Output path to where we want to save the computed
        metrics, defaults to "_results/coco_evaluation_results.json"
    """
    parsed_results = get_prediction_results(all_predictions_file_glob)

    print(f"Running evaluation on {len(parsed_results)} items")

    scores = evaluate(results=parsed_results, )

    # print out the results
    with open(output_path, 'w+') as f:
        json.dump(scores, f, indent=2)

    print()
    pp(scores)
    print()


if __name__ == '__main__':
    fire.Fire(cli_main)
