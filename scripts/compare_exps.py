import sys
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.embed import json_item
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import dodge
from omegaconf import OmegaConf
from torchvision.utils import save_image

sys.path.append(".")
from src.dataset.vtt import CATEGORIES, TOPICS, VTTDataset  # noqa: E402
from src.utils.datatool import read_json, read_jsonlines  # noqa: E402
from src.utils.plottool import matplotlib_header  # noqa: E402

dataset = VTTDataset(
    "test", return_raw_text=True, transform_cfg={"normalize": False}
)


LOG_ROOT = "/log/exp/vtt/"
details_cache = {}
exp_ids_cache = {}
results_cache = {}


def refresh():
    details_cache.clear()
    exp_ids_cache.clear()
    return gr.CheckboxGroup.update(choices=get_exp_ids())


def get_exp_ids():
    exp_names = []
    for exp_root in sorted(Path(LOG_ROOT).glob("*")):
        if (exp_root / "detail.jsonl").exists():
            config = OmegaConf.load(exp_root / "config.yaml")
            exp_name = config.name
            exp_id = exp_root.name
            exp_time = exp_id.split(".")[-1]
            i = 1
            while True:
                if (
                    exp_name in exp_ids_cache
                    and exp_ids_cache[exp_name] != exp_id
                ):
                    exp_name = f"{config.name}_{i}"
                    i += 1
                else:
                    break
            exp_ids_cache[exp_name] = exp_id
            exp_names.append((exp_name, exp_time))

            summary_path = (
                exp_root
                / "wandb"
                / "latest-run"
                / "files"
                / "wandb-summary.json"
            )
            if summary_path.exists():
                results_cache[exp_name] = read_json(summary_path)
    exp_names = [
        x[0] for x in sorted(exp_names, key=lambda x: x[1], reverse=True)
    ]
    return exp_names


def index2result(index, exp_names=[]):
    index = max(0, min(int(index), len(dataset) - 1))
    data = dataset[index]
    text_table = get_text_table(index, data, exp_names)
    metrics_plot = get_metrics_pyplot(index, exp_names)
    # metrics_plot = get_metrics_bokeh(index, exp_names)
    metrics_table = get_metrics_table(index, exp_names)
    pred_classification_table = get_classification_table(index, exp_names)
    overall_metrics_plot = get_overall_metrics_pyplot(exp_names)
    overall_metrics_table = get_overall_metrics_table(exp_names)
    return (
        index,
        CATEGORIES[data["category"]],
        TOPICS[data["topic"]],
        cache_test_image(data),
        text_table,
        metrics_plot,
        metrics_table,
        pred_classification_table,
        overall_metrics_plot,
        overall_metrics_table,
    )


def random_result(exp_names=[]):
    index = np.random.randint(len(dataset))
    return index2result(index, exp_names)


def cache_test_image(data, cache_dir="/data/vtt/cache/"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_image = cache_dir / f"test_{data['index']}.png"
    if not cache_image.exists():
        images = data["states"][data["states_mask"]]
        save_image(images, str(cache_image), nrow=images.size(0), pad_value=1.0)
    return str(cache_image)


def get_text_table(index, data, exp_names):
    results = {"NO.": list(range(1, len(data["text"]) + 1)), "GT": data["text"]}
    for exp_name in exp_names:
        if exp_name not in details_cache:
            exp_id = exp_ids_cache[exp_name]
            details_cache[exp_name] = read_jsonlines(
                f"{LOG_ROOT}/{exp_id}/detail.jsonl"
            )
        results[exp_name] = details_cache[exp_name][index]["preds"]
    df = pd.DataFrame(results)
    return df


def get_metrics_pyplot(index, exp_names):
    matplotlib_header(1 / 3)
    plt.rcParams["legend.fontsize"] = 12

    metrics = ["BLEU_4", "METEOR", "ROUGE", "CIDEr", "BERTScore"]
    fig = plt.figure()
    results = {
        key: [
            np.mean(details_cache[exp_name][index][key])
            for exp_name in exp_names
        ]
        for key in metrics
    }
    n_metrics = len(metrics)
    n_exp = len(exp_names)
    width = min((1 - 0.1) / n_exp, 0.2)

    x = np.arange(n_metrics)
    for i, exp_name in enumerate(exp_names):
        idx_exp = exp_names.index(exp_name)
        plt.bar(
            x + width * (i - n_exp / 2 + 0.5),
            [results[key][idx_exp] for key in metrics],
            width=width,
            label=exp_name,
        )
    plt.xticks(x, metrics)

    plt.legend()
    return fig


def get_metrics_bokeh(index, exp_names):
    metrics = ["BLEU_4", "METEOR", "ROUGE", "CIDEr", "BERTScore"]
    results = {
        "exp": exp_names,
    }
    results.update(
        {
            key: [
                np.mean(details_cache[exp_name][index][key])
                for exp_name in exp_names
            ]
            for key in metrics
        }
    )
    df = pd.DataFrame(results)
    source = ColumnDataSource(data=df)

    p = figure(
        x_range=exp_names,
        y_range=(0, 1),
        title="",
        height=350,
        toolbar_location=None,
        tools="",
    )

    n_metrics = len(metrics)
    width = min((1 - 0.1) / n_metrics, 0.2)
    for i, metric in enumerate(metrics):
        p.vbar(
            x=dodge("exp", (i - (n_metrics / 2)) * width, range=p.x_range),
            top=metric,
            width=width,
            source=source,
            legend_label=metric,
        )

    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"
    return json_item(p)


def get_classification_table(index, exp_names):
    keys = ["category_pred", "topic_pred"]
    results = {
        "Exp": exp_names,
    }
    results.update(
        {
            key: [
                details_cache[exp_name][index][key]
                if key in details_cache[exp_name][index]
                else " "
                for exp_name in exp_names
            ]
            for key in keys
        }
    )
    df = pd.DataFrame(results)
    return df


def get_metrics_table(index, exp_names):
    metrics = ["BLEU_4", "METEOR", "ROUGE", "CIDEr", "BERTScore"]
    results = {
        "Exp": exp_names,
    }
    results.update(
        {
            key: [
                np.mean(details_cache[exp_name][index][key])
                for exp_name in exp_names
            ]
            for key in metrics
        }
    )
    df = pd.DataFrame(results)
    return df


def get_overall_metrics_table(exp_names):
    metrics = ["BLEU_4", "METEOR", "ROUGE", "CIDEr", "BERTScore"]
    results = {
        "Exp": exp_names,
    }
    results.update(
        {
            key: [
                results_cache[exp_name][f"test/{key}"]
                if exp_name in results_cache
                and f"test/{key}" in results_cache[exp_name]
                else 0.0
                for exp_name in exp_names
            ]
            for key in metrics
        }
    )
    df = pd.DataFrame(results)
    return df


def get_overall_metrics_pyplot(exp_names):
    matplotlib_header(1 / 3)
    plt.rcParams["legend.fontsize"] = 12

    metrics = ["BLEU_4", "METEOR", "ROUGE", "CIDEr", "BERTScore"]
    fig = plt.figure()
    results = {
        key: [
            results_cache[exp_name][f"test/{key}"]
            if exp_name in results_cache
            and f"test/{key}" in results_cache[exp_name]
            else 0.0
            for exp_name in exp_names
        ]
        for key in metrics
    }
    n_metrics = len(metrics)
    n_exp = len(exp_names)
    width = min((1 - 0.1) / n_exp, 0.2)

    x = np.arange(n_metrics)
    for i, exp_name in enumerate(exp_names):
        idx_exp = exp_names.index(exp_name)
        plt.bar(
            x + width * (i - n_exp / 2 + 0.5),
            [results[key][idx_exp] for key in metrics],
            width=width,
            label=exp_name,
        )
    plt.xticks(x, metrics)

    plt.legend()
    return fig


with gr.Blocks(title="VTT") as demo:
    gr.Markdown("# Visual Transformation Telling")
    exp_id_input = gr.CheckboxGroup(choices=get_exp_ids(), label="Experiments")
    index_input = gr.Number(label="Image Index")
    with gr.Row():
        refresh_button = gr.Button("Refresh")
        random_button = gr.Button("Random")
        submit_button = gr.Button("View", variant="primary")
        random_button.style()

    with gr.Row():
        category = gr.Textbox(label="Category")
        topic = gr.Text(label="Topic")

    image_output = gr.Image()
    df_text_output = gr.DataFrame(label="Transformations")

    metrics_plot = gr.Plot(label="Metrics")
    metrics_df = gr.DataFrame(label="Metrics")
    classification_df = gr.DataFrame(label="Pred Classification")

    overall_metrics_plot = gr.Plot(label="Overall Metrics")
    overall_metrics_df = gr.DataFrame(label="Overall Metrics")

    refresh_button.click(refresh, inputs=None, outputs=exp_id_input)
    submit_button.click(
        index2result,
        inputs=[index_input, exp_id_input],
        outputs=[
            index_input,
            category,
            topic,
            image_output,
            df_text_output,
            metrics_plot,
            metrics_df,
            classification_df,
            overall_metrics_plot,
            overall_metrics_df,
        ],
    )
    random_button.click(
        random_result,
        inputs=exp_id_input,
        outputs=[
            index_input,
            category,
            topic,
            image_output,
            df_text_output,
            metrics_plot,
            metrics_df,
            classification_df,
            overall_metrics_plot,
            overall_metrics_df,
        ],
    )

demo.launch(server_name="0.0.0.0", share=False)
