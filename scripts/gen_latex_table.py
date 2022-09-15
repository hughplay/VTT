import argparse
import sys
from collections import defaultdict
from typing import Callable

import pandas as pd
import wandb


def main(args):

    api = wandb.Api()

    # api.default_entity by default
    entity = api.default_entity if args.entity is None else args.entity

    # get runs from the project
    def filter_runs(filters=None, sort=None):
        runs = api.runs(f"{entity}/{args.project}", filters=filters)
        runs = [
            run
            for run in runs
            if ("test/CIDEr" in run.summary and "model/_target_" in run.config)
        ]
        if sort is not None:
            runs = sorted(runs, key=sort)
        print(f"Find {len(runs)} runs in {entity}/{args.project}")
        return runs

    style, caption = getattr(sys.modules[__name__], f"{args.table}_table")(
        filter_runs
    )

    print()
    print(r"\usepackage{booktabs}")
    print()
    print(
        style.to_latex(
            caption=caption or args.caption,
            hrules=True,
            position=args.position,
            position_float="centering",
        )
    )


def main_table(filter_runs: Callable):
    filters = {"tags": {"$in": ["baseline"]}}
    runs = filter_runs(filters, sort=lambda run: run.summary["test/CIDEr"])
    results = defaultdict(list)
    for run in runs:
        model_name = run.config["model/_target_"].split(".")[-1]
        results["Model"].append(model_name)
        results["BLEU@4"].append(run.summary["test/BLEU_4"] * 100)
        results["METEOR"].append(run.summary["test/METEOR"] * 100)
        results["ROUGE-L"].append(run.summary["test/ROUGE"] * 100)
        results["CIDEr"].append(run.summary["test/CIDEr"] * 100)
        results["BERT-S"].append(run.summary["test/BERTScore"] * 100)

    df = pd.DataFrame(results)

    # generating the table by using: Pandas DataFrame.style.to_latex
    # https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_latex.html
    highlight_metrics = ["BLEU@4", "METEOR", "ROUGE-L", "CIDEr", "BERT-S"]
    style = df.style.highlight_max(
        axis=0, subset=highlight_metrics, props="textbf:--rwrap;"
    )
    style = style.format(precision=2).hide(axis="index")

    return style, "Model performance on the VTT dataset"


def classify_table(filter_runs: Callable):
    filters = {"tags": {"$in": ["multitask"]}}
    runs = filter_runs(filters, sort=lambda run: run.summary["test/CIDEr"])
    results = defaultdict(list)
    for run in runs:
        name = run.config["name"]
        if "no_classify" in name or "baseline" in name:
            name = "w/o category, topic"
        elif "no_category" in name:
            name = "w/o category"
        elif "no_topic" in name:
            name = "w/o topic"
        elif "category" in name:
            name = "w/ category"
        elif "topic" in name:
            name = "w/ topic"
        elif "reconstruct" in name:
            name = "w/ reconstruct"
        else:
            name = "w/ category, topic"

        if "sota" in run.config["name"]:
            name += " (SOTA)"

        results[" "].append(name)
        results["BLEU@4"].append(run.summary["test/BLEU_4"] * 100)
        results["METEOR"].append(run.summary["test/METEOR"] * 100)
        results["ROUGE-L"].append(run.summary["test/ROUGE"] * 100)
        results["CIDEr"].append(run.summary["test/CIDEr"] * 100)
        results["BERT-S"].append(run.summary["test/BERTScore"] * 100)

    df = pd.DataFrame(results)

    # generating the table by using: Pandas DataFrame.style.to_latex
    # https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_latex.html
    highlight_metrics = ["BLEU@4", "METEOR", "ROUGE-L", "CIDEr", "BERT-S"]
    style = df.style.highlight_max(
        axis=0, subset=highlight_metrics, props="textbf:--rwrap;"
    )
    style = style.format(precision=2).hide(axis="index")

    return style, "The effect of topic and category supervision"


def multiple_objectives_table(filter_runs: Callable):
    filters = {"tags": {"$in": ["sota_v4"]}}
    runs = filter_runs(filters)
    results = defaultdict(list)
    for run in runs:
        results["Model"].append(run.config["model/image_encoder"])
        checkmark = r"$\surd$"

        if "w_mtm" in run.config["name"]:
            results["MTM"].append(checkmark)
        else:
            results["MTM"].append("")
        if "w_category" in run.config["name"]:
            results["Category"].append(checkmark)
        else:
            results["Category"].append("")
        if "w_topic" in run.config["name"]:
            results["Topic"].append(checkmark)
        else:
            results["Topic"].append("")

        if "w_diff" in run.config["name"]:
            results["Diff"].append(checkmark)
        else:
            results["Diff"].append("")

        # results["name"].append(run.config["name"])
        results["BLEU@4"].append(run.summary["test/BLEU_4"] * 100)
        results["METEOR"].append(run.summary["test/METEOR"] * 100)
        results["ROUGE-L"].append(run.summary["test/ROUGE"] * 100)
        results["CIDEr"].append(run.summary["test/CIDEr"] * 100)
        results["BERT-S"].append(run.summary["test/BERTScore"] * 100)

    df = pd.DataFrame(results)
    df = df.sort_values(by=["Model", "CIDEr"])

    # generating the table by using: Pandas DataFrame.style.to_latex
    # https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_latex.html
    highlight_metrics = ["BLEU@4", "METEOR", "ROUGE-L", "CIDEr", "BERT-S"]
    style = df.style.highlight_max(
        axis=0, subset=highlight_metrics, props="textbf:--rwrap;"
    )
    style = style.format(precision=2).hide(axis="index")

    return style, "The effect of multiple objectives"


def image_encoder_table(filter_runs: Callable):

    filters = {"tags": {"$in": ["encoder"]}}
    runs = filter_runs(filters, sort=lambda run: run.summary["test/CIDEr"])

    name_mapping = {
        "inception_v3": "InceptionV3",
        "beit_large_patch16_224": "BEiT-L",
        "swin_large_patch4_window7_224": "Swin-L",
        "vit_large_patch16_224": "ViT-L",
        "resnet152": "ResNet152",
    }

    results = defaultdict(list)
    for run in runs:
        encoder_name = (
            run.config["model/image_encoder"]
            if "model/image_encoder" in run.config
            else "ViT-B/32"
        )
        if encoder_name in name_mapping:
            encoder_name = name_mapping[encoder_name]
        results["Image Encoder"].append(encoder_name)
        results["BLEU@4"].append(run.summary["test/BLEU_4"] * 100)
        results["METEOR"].append(run.summary["test/METEOR"] * 100)
        results["ROUGE-L"].append(run.summary["test/ROUGE"] * 100)
        results["CIDEr"].append(run.summary["test/CIDEr"] * 100)
        results["BERT-S"].append(run.summary["test/BERTScore"] * 100)

    df = pd.DataFrame(results)

    # generating the table by using: Pandas DataFrame.style.to_latex
    # https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_latex.html
    highlight_metrics = ["BLEU@4", "METEOR", "ROUGE-L", "CIDEr", "BERT-S"]
    style = df.style.highlight_max(
        axis=0, subset=highlight_metrics, props="textbf:--rwrap;"
    )
    style = style.format(precision=2).hide(axis="index")

    return style, "Performance of different image encoders on the VTT dataset"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=None)
    parser.add_argument("--table", default="main")
    parser.add_argument("--project", default="vtt")
    parser.add_argument(
        "--caption", default="Model performance on the VTT dataset."
    )
    parser.add_argument("--position", default="ht")
    args = parser.parse_args()
    main(args)
