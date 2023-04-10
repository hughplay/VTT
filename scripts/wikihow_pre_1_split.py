import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(".")
from src.dataset.text import SimpleTokenizer  # noqa: E402

random.seed(0)
DATA_ROOT = Path("/data/vtt/wikihow/")
MAX_TOKENS = 75  # all CLIP models use 77 as the context length, including the
# special tokens <|startoftext|> and <|endoftext|>


# data can be downloaded from https://github.com/mahnazkoupaee/WikiHow-Dataset
wikihow = pd.read_csv(DATA_ROOT / "wikihowAll.csv")


steps = set()
for headline in tqdm(wikihow.headline, ncols=80):
    try:
        step_list = headline.strip().split("\n")
        step_list = [step.strip(",") for step in step_list if step.strip(",")]
        for s in step_list:
            # remove "," at the beginning and end of the string
            s = s.strip(",").strip()
            # remove quotes at the beginning and end of the string
            s = s.strip('"').strip()
            s = s.strip("'").strip()
            if s:
                steps.add(s)
    except Exception as e:
        pass
steps = list(sorted(list(steps)))

with open(DATA_ROOT / "wikihow_steps.txt", "w") as f:
    f.write("\n".join(steps))


tokenizer = SimpleTokenizer()
samples_len = defaultdict(list)
for text in tqdm(steps, ncols=80):
    length = len(tokenizer.encode(text))
    if length <= MAX_TOKENS:
        samples_len[length].append(text)


split_ratio = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}
split_samples = defaultdict(list)
for length, samples in samples_len.items():
    random.shuffle(samples)
    start = 0
    for split, ratio in split_ratio.items():
        end = start + int(len(samples) * ratio)
        split_samples[split].extend(samples[start:end])
        start = end

validate_samples = set()
for split, samples in split_samples.items():
    print(f"{split}: {len(samples)}")
    random.shuffle(samples)
    with open(DATA_ROOT / f"wikihow_{split}.txt", "w") as f:
        f.write("\n".join(samples))

    for sample in samples:
        validate_samples.add(sample)

assert len(validate_samples) == sum(
    len(samples) for samples in split_samples.values()
)


# train: 1080563
# val: 135043
# test: 135043
