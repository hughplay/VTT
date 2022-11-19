import random
from pathlib import Path

import jsonlines
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tqdm import tqdm

META_FILE = Path("/data/vtt/meta/vtt.jsonl")
STATES_ROOT = Path("/data/vtt/states")
RANDOM_SAMPLE_ROOT = Path("/data/vtt/random_100")


def show_figures(path_list, title=None, labels=None, show_indices=True):
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
    fig, axarr = plt.subplots(1, n_img, figsize=(width * n_img, height))
    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    for i in range(n_img):
        # axarr[i].axis("off")
        if path_list[i].exists():
            axarr[i].imshow(mpimg.imread(path_list[i]))
            axarr[i].set_xticks([])
            axarr[i].set_yticks([])
            if show_indices:
                axarr[i].set_title(f"{i}")
            if labels is not None:
                axarr[i].set_xlabel(
                    "\n".join(wrap(f"{labels[i]}", width=width * 10))
                )

    # plt.subplots_adjust(hspace=0, wspace=0.05)
    if title:
        fig.suptitle(title)
    plt.tight_layout()


def show_sample(sample):
    n_states = len(sample["annotation"]) + 1
    t_states = [sample["annotation"][0]["segment"][0]] + [
        step["segment"][1] for step in sample["annotation"]
    ]
    state_path_list = [
        STATES_ROOT / f"{sample['id']}_{n_states}_{i}.jpg"
        for i in range(n_states)
    ]
    show_figures(
        state_path_list,
        title=f"{sample['ori']} - {sample['id']} - {sample['youtube_id']}",
        labels=[f"[ {t_states[0]} ]"]
        + [
            f"{s['label']} [ {t_states[i+1]} ]"
            for i, s in enumerate(sample["annotation"])
        ],
    )


def random_sample(n=100):
    with jsonlines.open(META_FILE) as reader:
        data = list(reader)

    RANDOM_SAMPLE_ROOT.mkdir(exist_ok=True, parents=True)
    random.seed(2023)
    for i in tqdm(range(n), ncols=80):
        sample = random.choice(data)
        show_sample(sample)
        plt.savefig(RANDOM_SAMPLE_ROOT / f"sample_{i:03d}_{sample['id']}.png")
        plt.close()


if __name__ == "__main__":
    random_sample(100)
