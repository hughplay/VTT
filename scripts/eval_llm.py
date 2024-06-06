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
    preds = {x["id"]: x["response"] for x in preds}

    n_sample_missing = 0
    n_transformation_missing = 0
    missing_ids = []

    processed_samples = []
    for sample in tqdm(targets, ncols=80):
        if sample["split"] != "test":
            continue
        id = sample["id"]
        n_transformation = len(sample["annotation"])
        transformations = [x["label"] for x in sample["annotation"]]

        pred_transformations = [""] * n_transformation
        if id not in preds:
            n_sample_missing += 1
        else:
            pred = extract_json(preds[id])
            if type(pred) is str:
                response_texts = pred.strip().split("\n")
                for trans in response_texts:
                    number, sentence = extract_number_and_sentence(trans)
                    if (
                        number is not None
                        and sentence is not None
                        and number <= n_transformation
                    ):
                        pred_transformations[number - 1] = sentence
            elif type(pred) is list:
                for i, trans in enumerate(pred):
                    if i < n_transformation:
                        pred_transformations[i] = pred[i]
        ttc.update(pred_transformations, transformations)
        # print(f"{len(transformations)}, {len(pred_transformations)}, {len(ttc.bleu_4.scores)}")

        n_missing = pred_transformations.count("")
        n_transformation_missing += n_missing
        if n_missing > 0:
            missing_ids.append(id)
        processed_samples.append(
            {
                "id": id,
                "transformations": pred_transformations,
            }
        )

    metrics = ttc.compute(verbose=True)

    for key, val in metrics.items():
        if type(val) is torch.Tensor:
            metrics[key] = val.item()
        print(f"{key}: {metrics[key]}")

    result = {
        "name": name,
        "metrics": metrics,
        "n_sample_missing": n_sample_missing,
        "n_transformation_missing": n_transformation_missing,
        "missing_ids": missing_ids,
    }
    return processed_samples, result, ttc.scores


def eval_wrapper(
    llm_output_path,
    gt_path,
    processed_path,
    result_path,
    bert_score_model="/data/pretrain/transformers/roberta-large",
    gpu=False,
):
    preds = read_jsonlines(llm_output_path)
    targets = read_jsonlines(gt_path)
    targets = [x for x in targets if x["split"] == "test"]
    processed_samples, result, scores = eval_vtt(
        Path(llm_output_path).stem,
        preds,
        targets,
        bert_score_model=bert_score_model,
        gpu=gpu,
    )
    write_jsonlines(processed_path, processed_samples)

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
    write_jsonlines(processed_path.replace(".jsonl", "_detail.jsonl"), details)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_output_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--processed_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument(
        "--bert_score_model",
        type=str,
        default="/data/pretrain/transformers/roberta-large",
    )
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    eval_wrapper(
        args.llm_output_path,
        args.gt_path,
        args.processed_path,
        args.result_path,
        args.bert_score_model,
        args.gpu,
    )
