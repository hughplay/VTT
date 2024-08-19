import datetime
import json
from pathlib import Path

import gradio as gr
import jsonlines
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).parent
TEST_SAMPLES_FILE = CURRENT_DIR / "vtt_test.jsonl"
TEST_RESULTS_FILE = CURRENT_DIR / "human_reason_samples.jsonl"
STATES_ROOT = Path("/data/vtt/states/")
IMAGE_CACHE_DIR = Path("/data/vtt/human_reason_cache")
IMAGE_CACHE_DIR.mkdir(exist_ok=True, parents=True)
RESULT_DIR = Path("/data/vtt/human_reason")
RESULT_DIR.mkdir(exist_ok=True, parents=True)
REPEAT = 1
MAX_IMAGES_ROW = 6


MODE = "EN"

if MODE == "EN":
    TITLE = "Human Evaluation for VTT"
    START_TEXT = "Start / Jump"
    NEXT_TEXT = "Next"
    SKIP_TEXT = "Cannot Decide"
    SUBMIT_TEXT = "Submit"
    FLUENCY_TEXT = "Fluency"
    RELEVANCE_TEXT = "Relevance"
    LOGICAL_SOUNDNESS_TEXT = "Logical Soundness"
    CATEGORY_TEXT = "Category"
    TOPIC_TEXT = "Topic"
    START_TIME_TEXT = "Start Time"
    LAST_COST_TIME_TEXT = "Last Sample Cost Time (s)"
    GUIDELINE_TEXT = "Guideline"
    COMPLETED_ANNOTATIONS_TEXT = "Completed Annotations"
    REFRESH_TEXT = "Refresh"
    TRANSFORMATIONS_TEXT = "Transformation Descriptions"
elif MODE == "CN":
    TITLE = "人工评估 VTT"
    START_TEXT = "开始 / 跳转"
    NEXT_TEXT = "下一个"
    SKIP_TEXT = "无法决定"
    SUBMIT_TEXT = "提交"
    FLUENCY_TEXT = "流畅度"
    RELEVANCE_TEXT = "相关性"
    LOGICAL_SOUNDNESS_TEXT = "逻辑性"
    CATEGORY_TEXT = "类别"
    TOPIC_TEXT = "主题"
    START_TIME_TEXT = "开始时间"
    LAST_COST_TIME_TEXT = "上一样本耗时 (秒)"
    GUIDELINE_TEXT = "指南"
    COMPLETED_ANNOTATIONS_TEXT = "已完成标注"
    REFRESH_TEXT = "刷新"
    TRANSFORMATIONS_TEXT = "变化描述"


with jsonlines.open(TEST_SAMPLES_FILE) as reader:
    samples = list(reader)
    samples_dict = {sample["id"]: sample for sample in samples}
with jsonlines.open(TEST_RESULTS_FILE) as reader:
    results = list(reader)

# chain repeated lists
results = [result for _ in range(REPEAT) for result in results]


def get_sample(annotation_id):
    validate_annotation_id(annotation_id)
    id = results[annotation_id]["id"]
    sample = samples_dict[id]
    return sample


def show_figures(path_list, title=None, show_indices=True):
    from textwrap import wrap

    n_img = len(path_list)
    width, height = plt.figaspect(1)

    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.linewidth"] = 0
    plt.rcParams["axes.titlepad"] = 6
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["axes.labelweight"] = "normal"
    plt.rcParams["font.size"] = 12
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 100
    plt.rcParams["figure.titlesize"] = 18

    # subplot(r,c) provide the no. of rows and columns

    if n_img > MAX_IMAGES_ROW:
        width = width / 2
        height = height / 2

    n_image_row = min(n_img, MAX_IMAGES_ROW)
    n_row = (n_img - 1) // n_image_row + 1
    fig, axarr = plt.subplots(
        n_row, n_image_row, figsize=(width * n_image_row, height * n_row)
    )
    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    for i in range(n_row * n_image_row):
        # axarr[i].axis("off")
        if n_row == 1:
            ax = axarr[i]
        else:
            ax = axarr[i // n_image_row][i % n_image_row]
        if i < len(path_list) and path_list[i].exists():
            ax.imshow(mpimg.imread(path_list[i]))
            if show_indices:
                ax.set_title(f"{i}")
        ax.set_xticks([])
        ax.set_yticks([])

    # plt.subplots_adjust(hspace=0, wspace=0.05)
    # if title:
    #     fig.suptitle(title)
    plt.tight_layout()


def show_sample(sample):
    n_states = len(sample["annotation"]) + 1
    state_path_list = [
        STATES_ROOT / f"{sample['id']}_{n_states}_{i}.jpg"
        for i in range(n_states)
    ]
    show_figures(state_path_list)


def get_image(annotation_id):
    sample = get_sample(annotation_id)
    cache_image = IMAGE_CACHE_DIR / f"human_evaluation_{annotation_id}.png"
    if not cache_image.exists():
        show_sample(sample)
        plt.savefig(cache_image)
        plt.close()
    return str(cache_image)


def save_result(annotation_id, info):
    result = results[annotation_id]
    result.update(info)
    with open(RESULT_DIR / f"human_{annotation_id}.json", "w") as f:
        json.dump(result, f)


def try_read_history(annotation_id):
    path = RESULT_DIR / f"human_{annotation_id}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return ", ".join(data["preds"])
    else:
        return None


def get_time_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def validate_annotation_id(annotation_id):
    annotation_id = max(0, min(int(annotation_id), len(results) - 1))
    return annotation_id


def start(annotation_id):
    annotation_id = validate_annotation_id(annotation_id)
    image = get_image(annotation_id)
    return (
        get_time_now(),
        image,
        try_read_history(annotation_id),
    )


def next_sample(annotation_id):
    annotation_id = validate_annotation_id(annotation_id + 1)
    image = get_image(annotation_id)
    return (
        annotation_id,
        get_time_now(),
        image,
        try_read_history(annotation_id),
    )


def submit(annotation_id, transformations, start_time):
    annotation_id = validate_annotation_id(annotation_id)
    time_now = get_time_now()
    duration = (
        datetime.datetime.strptime(time_now, "%Y-%m-%d %H:%M:%S")
        - datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    ).seconds
    save_result(
        annotation_id,
        {
            "start_time": start_time,
            "end_time": time_now,
            "duration": duration,
            "preds": [x.strip() for x in transformations.split(",")],
            "normal": True,
        },
    )
    annotation_id = validate_annotation_id(annotation_id + 1)
    image = get_image(annotation_id)
    return (
        annotation_id,
        get_time_now(),
        duration,
        image,
        try_read_history(annotation_id),
    )


def skip(annotation_id, start_time):
    annotation_id = validate_annotation_id(annotation_id)
    time_now = get_time_now()
    duration = (
        datetime.datetime.strptime(time_now, "%Y-%m-%d %H:%M:%S")
        - datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    ).seconds
    save_result(
        annotation_id,
        {
            "fluency": -1,
            "relevance": -1,
            "logical_soundness": -1,
            "start_time": start_time,
            "end_time": time_now,
            "duration": duration,
            "normal": False,
        },
    )
    annotation_id = validate_annotation_id(annotation_id + 1)
    image = get_image(annotation_id)
    return (
        annotation_id,
        get_time_now(),
        duration,
        image,
        try_read_history(annotation_id),
    )


def update_completed_annotations():
    path_list = list(RESULT_DIR.glob("human_*.json"))
    completed_annotations = sorted(
        [int(x.stem.split("_")[1]) for x in path_list]
    )
    complete_str = ""
    pre = None
    tmp_str = ""
    for x in completed_annotations:
        if pre is None:
            complete_str += f"{x}"
        elif x > pre + 1:
            complete_str += tmp_str
            complete_str += f", {x}"
            tmp_str = ""
        else:
            tmp_str = f"-{x}"
        pre = x
    complete_str += tmp_str

    return complete_str


def main():

    with gr.Blocks(title="VTT") as demo:
        gr.Markdown(f"## {TITLE}")

        with gr.Row():
            with gr.Column():
                annotation_id = gr.Number(label="Annotation ID")
                with gr.Row():
                    start_button = gr.Button(START_TEXT)
                    next_button = gr.Button(NEXT_TEXT)

        image = gr.Image()

        transformations = gr.Text(label=TRANSFORMATIONS_TEXT)

        with gr.Row():
            skip_button = gr.Button(SKIP_TEXT)
            submit_button = gr.Button(SUBMIT_TEXT, variant="primary")

        with gr.Row():
            start_time = gr.Text(label=START_TIME_TEXT)
            last_duration = gr.Text(label=LAST_COST_TIME_TEXT)

        with gr.Tab(COMPLETED_ANNOTATIONS_TEXT):
            refresh_button = gr.Button(REFRESH_TEXT)
            completed_annotations = gr.Text(label=COMPLETED_ANNOTATIONS_TEXT)

        refresh_button.click(
            update_completed_annotations, outputs=[completed_annotations]
        )

        start_button.click(
            start,
            inputs=[annotation_id],
            outputs=[
                start_time,
                image,
                transformations,
            ],
        )
        next_button.click(
            next_sample,
            inputs=[annotation_id],
            outputs=[
                annotation_id,
                start_time,
                image,
                transformations,
            ],
        )
        skip_button.click(
            skip,
            inputs=[
                annotation_id,
                start_time,
            ],
            outputs=[
                annotation_id,
                start_time,
                last_duration,
                image,
                transformations,
            ],
        )
        submit_button.click(
            submit,
            inputs=[
                annotation_id,
                transformations,
                start_time,
            ],
            outputs=[
                annotation_id,
                start_time,
                last_duration,
                image,
                transformations,
            ],
        )

    # demo.launch(server_name="0.0.0.0", share=True)
    demo.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    main()
