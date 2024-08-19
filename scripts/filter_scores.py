import argparse
import sys
from collections import defaultdict

import numpy as np

sys.path.append(".")  # noqa: E402
from src.utils.datatool import read_jsonlines  # noqa: E402

automatic_metrics = [
    "BLEU_4",
    "METEOR",
    "ROUGE",
    "CIDEr",
    "SPICE",
    "BERTScore",
    "fluency",
    "relevance",
    "logical_soundness",
]


def filter_scores(ref, target):
    ref_data = read_jsonlines(ref)
    target_data = read_jsonlines(target)

    ref_ids = set([x["id"] for x in ref_data])
    target_samples = [x for x in target_data if x["id"] in ref_ids]

    print(f"total ref: {len(ref_data)}")
    print(f"total target: {len(target_samples)}")

    metrics = defaultdict(list)
    for sample in target_samples:
        for metric in automatic_metrics:
            if type(sample[metric]) is list:
                metrics[metric].extend(sample[metric])
            else:
                metrics[metric].append(sample[metric])

    for metric, scores in metrics.items():
        print(f"{metric}: {np.mean(scores)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    args = parser.parse_args()
    filter_scores(args.ref, args.target)
