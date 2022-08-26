import argparse
import sys
from collections import defaultdict

import pandas as pd
import wandb


def main(args):

    api = wandb.Api()

    # api.default_entity by default
    entity = api.default_entity if args.entity is None else args.entity

    # get runs from the project
    runs = api.runs(f"{entity}/{args.project}")
    print(f"Find {len(runs)} runs in {entity}/{args.project}")

    style = getattr(sys.modules[__name__], f"{args.table}_table")(runs)

    print()
    print(r"\usepackage{booktabs}")
    print()
    print(
        style.to_latex(
            caption=args.caption,
            hrules=True,
            position=args.position,
            position_float="centering",
        )
    )


def main_table(runs):
    # prepare table content
    results = defaultdict(list)
    for run in runs:
        if "test/ROUGE" not in run.summary:
            continue
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

    return style


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
