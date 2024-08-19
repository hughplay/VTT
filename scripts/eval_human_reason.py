import argparse
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.append(".")  # noqa: E402
from src.criterion.tell import TTCriterion  # noqa: E402
from src.utils.datatool import read_jsonlines, write_jsonlines  # noqa: E402


def extract_json(text):
    try:
        text = text.strip()
        if text.startswith("{") or text.startswith("```"):
            text = text.replace("```json", "")
            text = text.replace("```", "")
            text = eval(text)
            text = text["transformations"]
    except Exception as e:
        print(f"Error: {e}, text: {text}")
        return text
    return text


def extract_number_and_sentence(sentence):
    pattern = r"^(\d+)\.\s*(.*)"
    match = re.match(pattern, sentence)
    if match:
        number = int(match.group(1))
        sentence_without_number = match.group(2).strip()
        return number, sentence_without_number
    else:
        return None, None


def eval_vtt(
    name,
    preds,
    targets,
    bert_score_model="/data/pretrain/transformers/roberta-large",
    gpu=False,
):
    ttc = TTCriterion(bert_score_model=bert_score_model)
    if gpu:
        ttc = ttc.cuda()

    for pred, target in tqdm(zip(preds, targets), ncols=80):
        pred_transformations = pred["preds"]
        transformations = [x["label"] for x in target["annotation"]]
        assert len(transformations) == len(pred_transformations)
        ttc.update(pred_transformations, transformations)

    metrics = ttc.compute(verbose=True)

    for key, val in metrics.items():
        if type(val) is torch.Tensor:
            metrics[key] = val.item()
        print(f"{key}: {metrics[key]}")

    result = {
        "name": name,
        "metrics": metrics,
    }
    return result, ttc.scores


def eval_wrapper(
    pred_path,
    gt_path,
    result_path,
    detail_path,
    bert_score_model="/data/pretrain/transformers/roberta-large",
    gpu=False,
):
    preds = read_jsonlines(pred_path)
    targets = {x["id"]: x for x in read_jsonlines(gt_path)}
    targets = [targets[x["id"]] for x in preds]
    result, scores = eval_vtt(
        Path(pred_path).stem,
        preds,
        targets,
        bert_score_model=bert_score_model,
        gpu=gpu,
    )

    results = []
    if Path(result_path).exists():
        results = read_jsonlines(result_path)
    results.append(result)
    write_jsonlines(result_path, results)

    total_seq = sum([len(x["annotation"]) for x in targets])
    for _, score_list in scores.items():
        assert len(score_list) == total_seq, f"{len(score_list)} != {total_seq}"
    details = []
    curr_idx = 0
    for sample in targets:
        n_transformation = len(sample["annotation"])
        sample_metrics = {}
        for metric, score_list in scores.items():
            sample_metrics[metric] = score_list[
                curr_idx : curr_idx + n_transformation
            ]
        details.append(
            {
                "id": sample["id"],
                "metrics": sample_metrics,
            }
        )
        curr_idx += n_transformation
    write_jsonlines(detail_path, details)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--detail_path", type=str, required=True)
    parser.add_argument(
        "--bert_score_model",
        type=str,
        default="/data/pretrain/transformers/roberta-large",
    )
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    eval_wrapper(
        args.pred_path,
        args.gt_path,
        args.result_path,
        args.detail_path,
        args.bert_score_model,
        args.gpu,
    )
