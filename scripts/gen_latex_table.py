import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import pandas as pd
import wandb

CHECKMARK = r"$\surd$"
RENAME = {
    "BLEU@4": "B@4",
    "METEOR": "M",
    "ROUGE-L": "R",
    "CIDEr": "C",
    "SPICE": "S",
    "BERT-S": "BS",
}

# https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv
IMAGENET_ACC = {
    "InceptionV3": 77.438,
    "ResNet152": 82.818,
    "BEiT-L": 87.476,
    "Swin-L": 86.320,
    "ViT-L": 85.844,
    "RN50": 73.3,
    "RN101": 75.7,
    "ViT-B/32": 76.1,
    "ViT-B/16": 80.2,
    "ViT-L/14": 83.9,
}


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
    style, caption, position="ht", small=True, save_path=None, **kwargs
):
    print(r"\usepackage{booktabs}")
    print()
    # style.applymap_index(lambda v: "font-weight: bold;", axis="index")
    # style.applymap_index(lambda v: "font-weight: bold;", axis="columns")
    latex_str = style.to_latex(
        caption=caption,
        hrules=True,
        position=position,
        position_float="centering",
        # convert_css=True,
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
            context_encoder = "Transformer"
        else:
            context_encoder = "LSTM"
        # results["$f_{context}$"].append(
        #     f"{context_encoder} ({run.summary['model_size/context_encoder'] / 4:.0f}M)"
        # )

        if "TTNet" in model_name:
            text_decoder = "Transformer"
        else:
            text_decoder = "LSTM"
        # results["$f_{decoder}$"].append(
        #     f"{text_decoder} ({run.summary['model_size/decoder'] / 4:.0f}M)"
        # )

        if "TTNet" in model_name:
            if model_name == "TTNetDiff":
                model_name = "TTNet"
            else:
                model_name = "TTNet$_\\text{Base}$"
        elif image_encoder == "CLIP":
            model_name += "*"

        results["Model"].append(model_name)

        results["Architecture"].append(
            f"{image_encoder} / {context_encoder} / {text_decoder}"
        )
        results["Params"].append(f"{run.summary['model_size/total'] / 4:.0f}M")

        results["BLEU@4"].append(run.summary["test/BLEU_4"] * 100)
        results["METEOR"].append(run.summary["test/METEOR"] * 100)
        results["ROUGE-L"].append(run.summary["test/ROUGE"] * 100)
        results["CIDEr"].append(run.summary["test/CIDEr"] * 100)
        results["SPICE"].append(run.summary["test/SPICE"] * 100)
        results["BERT-S"].append(run.summary["test/BERTScore"] * 100)

    df = pd.DataFrame(results)

    df.rename(columns=RENAME, inplace=True)
    highlight_metrics = list(RENAME.values())
    style = df.style.highlight_max(
        axis=0, subset=highlight_metrics, props="textbf:--rwrap;"
    )
    style = style.format(precision=2).hide(axis="index")

    str_latex = gen_latex(
        style,
        "Performance on the test set of VTT dataset. "
        "B@4/M/R/C/S/BS are short for "
        "BLEU@4 / METEOR / ROUGE-L / CIDEr / SPICE / BERT-Score. "
        "The architecture shows image encoder / context encoder "
        "/ transformation decoder. * indicates to use CLIP "
        "for a fair comparison.",
        save_path="docs/tables/baseline.tex",
        label="tab:baseline",
        column_format="lrrrrrrrr",
        position="t",
    )

    lines = str_latex.split("\n")
    new_lines = []
    for line in lines:
        if line.startswith("TTNet$"):
            new_lines.append("\\midrule")
        new_lines.append(line)
        if line.startswith("\\label"):
            new_lines.append("\\setlength{\\tabcolsep}{4pt}")
    str_latex = "\n".join(new_lines)
    return str_latex


def image_encoder_table(filter_runs: Callable):

    filters = {
        # "tags": {"$in": ["encoder"]},
        "$and": [
            {"tags": {"$in": ["sota_v5"]}},
            {"tags": {"$in": ["encoder"]}},
        ],
    }
    runs = filter_runs(filters, sort=lambda run: run.summary["test/CIDEr"])

    name_mapping = {
        "inception_v3": "InceptionV3",
        "beit_large_patch16_224": "BEiT-L",
        "swin_large_patch4_window7_224": "Swin-L",
        "vit_large_patch16_224": "ViT-L",
        "resnet152": "ResNet152",
    }

    results = defaultdict(list)
    encoders = []
    pretrained = []
    for run in runs:
        encoder_name = (
            run.config["model/image_encoder"]
            if "model/image_encoder" in run.config
            else "ViT-B/32"
        )
        if encoder_name in name_mapping:
            encoder_name = name_mapping[encoder_name]
        # results["Image Encoder"].append(encoder_name)
        if "/" in encoder_name or encoder_name.startswith("RN"):
            pretrained.append(
                "\\rotatebox[origin=c]{90}{\\makecell"
                "{Image-text\\\\Pretrained\\footnotemark[2]}}"
            )
        else:
            pretrained.append(
                "\\rotatebox[origin=c]{90}{\\makecell"
                "{ImageNet\\\\Pretrained\\footnotemark[1]}}"
            )
        encoders.append(encoder_name)
        results["Params"].append(
            f"{run.summary['model_size/image_encoder'] / 4:.0f}M"
        )
        results["Acc"].append(IMAGENET_ACC[encoder_name])
        results["BLEU@4"].append(run.summary["test/BLEU_4"] * 100)
        results["METEOR"].append(run.summary["test/METEOR"] * 100)
        results["ROUGE-L"].append(run.summary["test/ROUGE"] * 100)
        results["CIDEr"].append(run.summary["test/CIDEr"] * 100)
        results["BERT-S"].append(run.summary["test/BERTScore"] * 100)
        results["pretrained"].append(pretrained[-1])
        results["param"].append(run.summary["model_size/total"] / 4)

    df = pd.DataFrame(results, index=[pretrained, encoders])
    df = df.sort_values(by=["pretrained", "param"], ascending=[False, True])
    df = df.drop(columns=["pretrained", "param"])
    df = df.drop(columns=["METEOR", "ROUGE-L"])

    df.rename(columns=RENAME, inplace=True)
    highlight_metrics = [
        val for val in list(RENAME.values()) if val in df.columns
    ]
    style = df.style.highlight_max(
        axis=0, subset=highlight_metrics, props="textbf:--rwrap;"
    )
    style = style.format(precision=2)

    str_latex = gen_latex(
        style,
        "Performance of different image encoders on the VTT dataset",
        label="tab:image_encoder",
        save_path="docs/tables/image_encoder.tex",
        column_format="llrr|rrr",
    )

    lines = str_latex.split("\n")
    new_lines = []
    for line in lines:
        if "RN50" in line:
            new_lines.append("\\midrule")
        elif "Params" in line:
            columns = line.split("&")
            columns = ["\\multicolumn{2}{c}{Image Encoder}"] + columns[2:]
            line = "&".join(columns)
        new_lines.append(line)
    str_latex = "\n".join(new_lines)
    # str_latex = str_latex.replace("caption", "captionof{table}")
    # str_latex = str_latex.replace(r"\begin{table}[ht]", r"\begin{minipage}[c]{0.5\textwidth}")
    # str_latex = str_latex.replace(r"\end{table}", r"\end{minipage}")
    return str_latex


def key_table(filter_runs: Callable):
    filters = {"tags": {"$in": ["key"]}}
    runs = filter_runs(filters)
    results = defaultdict(list)
    checkmark = r"$\surd$"
    for run in runs:
        if "model/diff_mode" in run.config:
            results["Diff."].append(checkmark)
        else:
            results["Diff."].append("")

        if (
            "model/mask_ratio" in run.config
            and run.config["model/mask_ratio"] > 0
        ):
            results["MTM"].append(checkmark)
        else:
            results["MTM"].append("")

        if (
            "criterion/loss/w_classify" in run.config
            and run.config["criterion/loss/w_classify"] is not None
            and run.config["criterion/loss/w_classify"] != "None"
        ):
            results["Aux."].append(checkmark)
        else:
            results["Aux."].append("")

        results["BLEU@4"].append(run.summary["test/BLEU_4"] * 100)
        results["METEOR"].append(run.summary["test/METEOR"] * 100)
        results["ROUGE-L"].append(run.summary["test/ROUGE"] * 100)
        results["CIDEr"].append(run.summary["test/CIDEr"] * 100)
        results["BERT-S"].append(run.summary["test/BERTScore"] * 100)

    df = pd.DataFrame(results)
    df["n_key"] = (df == checkmark).sum(axis=1)
    df.sort_values(
        by=["n_key", "Diff.", "MTM", "Aux."],
        ascending=[True, False, False, False],
        inplace=True,
    )
    df = df[
        [
            "Diff.",
            "MTM",
            "Aux.",
            "BLEU@4",
            "METEOR",
            "ROUGE-L",
            "CIDEr",
            "BERT-S",
        ]
    ]

    df = df.drop(columns=["METEOR", "ROUGE-L"])

    df.rename(columns=RENAME, inplace=True)
    highlight_metrics = [
        val for val in list(RENAME.values()) if val in df.columns
    ]
    style = df.style.highlight_max(
        axis=0, subset=highlight_metrics, props="textbf:--rwrap;"
    )
    style = style.format(precision=2).hide(axis="index")

    str_latex = gen_latex(
        style,
        "Ablation studies of key components.",
        save_path="docs/tables/key.tex",
        label="tab:key",
    )
    # str_latex = str_latex.replace("caption", "captionof{table}")
    # str_latex = str_latex.replace(r"\begin{table}[ht]", r"\begin{minipage}[c]{0.5\textwidth}")
    # str_latex = str_latex.replace(r"\end{table}", r"\end{minipage}")
    return str_latex


def diff_table(filter_runs: Callable):
    filters = {
        "$and": [
            {"tags": {"$in": ["sota_v5"]}},
            {"tags": {"$in": ["diff"]}},
        ],
    }
    runs = filter_runs(filters)
    results = defaultdict(list)
    for run in runs:
        if "model/diff_only" in run.config and run.config["model/diff_only"]:
            results["State"].append("")
        else:
            results["State"].append(CHECKMARK)

        if "model/diff_mode" in run.config:
            if run.config["model/diff_mode"] == "early":
                results["Difference"].append("early")
            elif run.config["model/diff_mode"] == "late":
                results["Difference"].append("late")
            elif run.config["model/diff_mode"] == "early_and_late":
                results["Difference"].append("early + late")
            else:
                results["Difference"].append("None")
        else:
            results["Difference"].append("")

        results["BLEU@4"].append(run.summary["test/BLEU_4"] * 100)
        results["METEOR"].append(run.summary["test/METEOR"] * 100)
        results["ROUGE-L"].append(run.summary["test/ROUGE"] * 100)
        results["CIDEr"].append(run.summary["test/CIDEr"] * 100)
        results["BERT-S"].append(run.summary["test/BERTScore"] * 100)

    df = pd.DataFrame(results)
    df = df.sort_values(by=["CIDEr"])

    df = df.drop(columns=["METEOR", "ROUGE-L"])

    df.rename(columns=RENAME, inplace=True)
    highlight_metrics = [
        val for val in list(RENAME.values()) if val in df.columns
    ]
    style = df.style.highlight_max(
        axis=0, subset=highlight_metrics, props="textbf:--rwrap;"
    )
    style = style.format(precision=2).hide(axis="index")

    return gen_latex(
        style,
        "Ablation study on difference features.",
        label="tab:diff",
        save_path="docs/tables/diff.tex",
    )


def mask_table(filter_runs: Callable):
    filters = {
        "$and": [
            {"tags": {"$in": ["sota_v5"]}},
            {"tags": {"$in": ["mask"]}},
        ],
    }
    runs = filter_runs(filters)
    results = defaultdict(list)
    for run in runs:
        mask_ratio = run.config["model/mask_ratio"]
        if mask_ratio < 0:
            mask_ratio = 0.0
        results["Mask Ratio"].append(f"{mask_ratio * 100:.0f}%")

        results["BLEU@4"].append(run.summary["test/BLEU_4"] * 100)
        results["METEOR"].append(run.summary["test/METEOR"] * 100)
        results["ROUGE-L"].append(run.summary["test/ROUGE"] * 100)
        results["CIDEr"].append(run.summary["test/CIDEr"] * 100)
        results["BERT-S"].append(run.summary["test/BERTScore"] * 100)

    df = pd.DataFrame(results)
    df = df.sort_values(by=["Mask Ratio"])

    df = df.drop(columns=["METEOR", "ROUGE-L"])

    df.rename(columns=RENAME, inplace=True)
    highlight_metrics = [
        val for val in list(RENAME.values()) if val in df.columns
    ]
    style = df.style.highlight_max(
        axis=0, subset=highlight_metrics, props="textbf:--rwrap;"
    )
    style = style.format(precision=2).hide(axis="index")

    return gen_latex(
        style,
        "Ablation on the mask ratio.",
        label="tab:mask_ratio",
        save_path="docs/tables/mask_ratio.tex",
        column_format="crrr",
        position="t",
    )


def mask_sample_table(filter_runs: Callable):
    filters = {
        "$and": [
            {"tags": {"$in": ["sota_v5"]}},
            {"tags": {"$in": ["sample_mask"]}},
        ],
    }
    runs = filter_runs(filters)
    results = defaultdict(list)
    for run in runs:
        if run.config["model/mask_ratio"] < 0:
            sample_mask_prob = 0.0
        else:
            sample_mask_prob = run.config["model/sample_mask_prob"]
        results["Sample Ratio"].append(f"{sample_mask_prob*100:.0f}%")

        results["BLEU@4"].append(run.summary["test/BLEU_4"] * 100)
        results["METEOR"].append(run.summary["test/METEOR"] * 100)
        results["ROUGE-L"].append(run.summary["test/ROUGE"] * 100)
        results["CIDEr"].append(run.summary["test/CIDEr"] * 100)
        results["BERT-S"].append(run.summary["test/BERTScore"] * 100)

    df = pd.DataFrame(results)
    df = df.sort_values(by=["Sample Ratio"])

    df = df.drop(columns=["METEOR", "ROUGE-L"])

    df.rename(columns=RENAME, inplace=True)
    highlight_metrics = [
        val for val in list(RENAME.values()) if val in df.columns
    ]
    style = df.style.highlight_max(
        axis=0, subset=highlight_metrics, props="textbf:--rwrap;"
    )
    style = style.format(precision=2).hide(axis="index")

    return gen_latex(
        style,
        "Ablation on sample ratio.",
        label="tab:mask_sample_ratio",
        save_path="docs/tables/mask_sample_ratio.tex",
        column_format="crrr",
        position="t",
    )


def classify_table(filter_runs: Callable):
    filters = {
        "$and": [
            {"tags": {"$in": ["sota_v5"]}},
            {"tags": {"$in": ["classify"]}},
        ],
    }
    runs = filter_runs(filters, sort=lambda run: run.summary["test/CIDEr"])
    results = defaultdict(list)
    exps = []
    for run in runs:
        if (
            "criterion/loss/w_category" in run.config
            and run.config["criterion/loss/w_category"] > 0
        ):
            results["Category"].append(CHECKMARK)
        else:
            results["Category"].append("")
        if (
            "criterion/loss/w_topic" in run.config
            and run.config["criterion/loss/w_topic"] > 0
        ):
            results["Topic"].append(CHECKMARK)
        else:
            results["Topic"].append("")
        results["BLEU@4"].append(run.summary["test/BLEU_4"] * 100)
        results["METEOR"].append(run.summary["test/METEOR"] * 100)
        results["ROUGE-L"].append(run.summary["test/ROUGE"] * 100)
        results["CIDEr"].append(run.summary["test/CIDEr"] * 100)
        results["BERT-S"].append(run.summary["test/BERTScore"] * 100)

    df = pd.DataFrame(results)

    df = df.drop(columns=["METEOR", "ROUGE-L"])

    df.rename(columns=RENAME, inplace=True)
    highlight_metrics = [
        val for val in list(RENAME.values()) if val in df.columns
    ]
    style = df.style.highlight_max(
        axis=0, subset=highlight_metrics, props="textbf:--rwrap;"
    )
    style = style.format(precision=2).hide(axis="index")

    return gen_latex(
        style,
        "The effect of topic and category supervision",
        label="tab:classify",
        save_path="docs/tables/classify.tex",
    )


def single_table(filter_runs: Callable):
    filters = {"tags": {"$in": ["single"]}}
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
            context_encoder = "Transformer"
        else:
            context_encoder = "LSTM"
        # results["$f_{context}$"].append(
        #     f"{context_encoder} ({run.summary['model_size/context_encoder'] / 4:.0f}M)"
        # )

        if "TTNet" in model_name:
            text_decoder = "Transformer"
        else:
            text_decoder = "LSTM"
        # results["$f_{decoder}$"].append(
        #     f"{text_decoder} ({run.summary['model_size/decoder'] / 4:.0f}M)"
        # )

        if "TTNet" in model_name:
            if model_name == "TTNetDiff":
                model_name = "TTNet"
            else:
                model_name = "TTNet w/o diff"
        elif image_encoder == "CLIP":
            model_name += "*"

        results["Model"].append(model_name)

        results["Original"].append(run.summary["test/CIDEr"] * 100)
        # results["Original/BERT-S"].append(run.summary["test/BERTScore"] * 100)
        results["Independent"].append(run.summary["single_test/CIDEr"] * 100)
        # results["Independent/BERT-S"].append(run.summary["single_test/BERTScore"] * 100)

    # models = results["Model"]
    # del results["Model"]

    # df = pd.DataFrame(results, index=models)
    df = pd.DataFrame(results)
    # df.columns = pd.MultiIndex.from_tuples(
    #     [(col.split("/")[0], col.split("/")[1]) for col in df.columns]
    # )

    # df.rename(columns=RENAME, inplace=True)
    # highlight_metrics = [
    #     val for val in list(RENAME.values()) if val in df.columns
    # ]
    style = df.style.highlight_max(
        axis=0, subset=["Original", "Independent"], props="textbf:--rwrap;"
    )
    style = style.format(precision=2).hide(axis="index")

    str_latex = gen_latex(
        style,
        "CIDEr of independent transformation prediction.",
        save_path="docs/tables/single.tex",
        label="tab:single",
    )

    return str_latex


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=None)
    parser.add_argument("--table", default="main")
    parser.add_argument("--project", default="vtt")
    parser.add_argument(
        "--caption", default="Model performance on the VTT dataset."
    )
    args = parser.parse_args()
    main(args)
