import argparse
import sys
from collections import defaultdict
from pathlib import Path
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

    latex_str = getattr(sys.modules[__name__], f"{args.table}_table")(
        filter_runs
    )
    print(latex_str)


def gen_latex(
    style, caption, position="ht", small=False, save_path=None, **kwargs
):
    print(r"\usepackage{booktabs}")
    print()
    latex_str = style.to_latex(
        caption=caption,
        hrules=True,
        position=position,
        position_float="centering",
        **kwargs,
    )
    if small:
        latex_str = latex_str.replace("\\centering", "\\centering\n\\small")
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as f:
            f.write(latex_str)
    return latex_str


def baseline_table(filter_runs: Callable):
    filters = {"tags": {"$in": ["final_base"]}}
    runs = filter_runs(filters, sort=lambda run: run.summary["test/CIDEr"])
    results = defaultdict(list)
    for run in runs:
        model_name = run.config["model/_target_"].split(".")[-1]

        if "model/image_encoder" in run.config:
            image_encoder = run.config["model/image_encoder"]
        else:
            image_encoder = "ResNet152"
        image_encoder = image_encoder.replace("resnet", "ResNet")
        if image_encoder == "ViT-L/14":
            image_encoder = "CLIP"
        elif image_encoder == "inception_v3":
            image_encoder = "InceptionV3"
        # results["$f_{image}$"].append(
        #     f"{image_encoder} ({run.summary['model_size/image_encoder'] / 4:.0f}M)"
        # )

        if "TTNet" in model_name:
            context_encoder = "T"
        else:
            context_encoder = "L"
        # results["$f_{context}$"].append(
        #     f"{context_encoder} ({run.summary['model_size/context_encoder'] / 4:.0f}M)"
        # )

        if "TTNet" in model_name:
            text_decoder = "T"
        else:
            text_decoder = "L"
        # results["$f_{decoder}$"].append(
        #     f"{text_decoder} ({run.summary['model_size/decoder'] / 4:.0f}M)"
        # )

        if "TTNet" in model_name:
            if model_name == "TTNetDiff":
                model_name = "TTNet"
            else:
                model_name = "TTNet$_\\text{base}$"
        elif image_encoder == "CLIP":
            model_name += "+"

        results["Model"].append(model_name)

        results["Architecture"].append(
            f"{image_encoder} / {context_encoder} / {text_decoder}"
        )
        results["Params"].append(f"{run.summary['model_size/total'] / 4:.0f}M")

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

    str_latex = gen_latex(
        style,
        "Performance on the test set of VTT dataset. "
        "T indicates Transformer, L indicates LSTM.",
        save_path="docs/tables/baseline.tex",
        small=True,
        label="tab:baseline",
    )

    lines = str_latex.split("\n")
    new_lines = []
    for line in lines:
        if line.startswith("TTNet*"):
            new_lines.append("\\midrule")
        new_lines.append(line)
    str_latex = "\n".join(new_lines)
    return str_latex


def base_table(filter_runs: Callable):
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

    return gen_latex(style, "Model performance on the VTT dataset")


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

    return gen_latex(style, "The effect of topic and category supervision")


def diff_table(filter_runs: Callable):
    runs = filter_runs()
    results = defaultdict(list)
    for run in runs:
        if (
            "ttnet_diff" in run.config["name"]
            or run.config["name"] == "baseline_ttnet_context_add"
        ):
            if "late" in run.config["name"]:
                results["Diff"].append("Late")
            elif "early" in run.config["name"]:
                results["Diff"].append("Early")
            elif "both" in run.config["name"]:
                results["Diff"].append("Both")
            else:
                results["Diff"].append("-")

            if "attention" in run.config["name"]:
                results["Context Fusion"].append("Attention")
            elif "cross" in run.config["name"]:
                results["Context Fusion"].append("Cross Attention")
            elif "fuse" in run.config["name"]:
                results["Context Fusion"].append("Linear Projection")
            else:
                results["Context Fusion"].append("-")

            if "first" in run.config["name"]:
                results["Query"].append("difference")
            elif "last" in run.config["name"]:
                results["Query"].append("states")
            else:
                results["Query"].append("-")

            results["BLEU@4"].append(run.summary["test/BLEU_4"] * 100)
            results["METEOR"].append(run.summary["test/METEOR"] * 100)
            results["ROUGE-L"].append(run.summary["test/ROUGE"] * 100)
            results["CIDEr"].append(run.summary["test/CIDEr"] * 100)
            results["BERT-S"].append(run.summary["test/BERTScore"] * 100)

    df = pd.DataFrame(results)
    df = df.sort_values(by=["CIDEr"])

    # generating the table by using: Pandas DataFrame.style.to_latex
    # https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_latex.html
    highlight_metrics = ["BLEU@4", "METEOR", "ROUGE-L", "CIDEr", "BERT-S"]
    style = df.style.highlight_max(
        axis=0, subset=highlight_metrics, props="textbf:--rwrap;"
    )
    style = style.format(precision=2).hide(axis="index")

    return gen_latex(style, "The effect of difference features")


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

    return gen_latex(style, "The effect of multiple objectives")


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

    return gen_latex(
        style, "Performance of different image encoders on the VTT dataset"
    )


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
