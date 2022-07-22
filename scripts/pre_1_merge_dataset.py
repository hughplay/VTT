import hashlib
import json
from pathlib import Path

import jsonlines

CROSS_TASK_ROOT = Path("/data/reason/CrossTask/crosstask_release/")
VAR_ROOT = Path("/data/reason/VAR")
COIN_ROOT = Path("/data/reason/coin")
VTT_ROOT = Path("/data/reason/vtt")


def str2hashkey(s, length=16):
    return hashlib.md5(s.encode()).hexdigest()[:length]


def is_steps_overlap(steps):
    for i in range(len(steps) - 1):
        if steps[i]["segment"][1] > steps[i + 1]["segment"][0]:
            return True
    return False


def cross2vtt():
    samples = []
    keys = set()
    key_repeat = 0
    with jsonlines.open(CROSS_TASK_ROOT / "tasks.jsonl") as reader:
        tasks = list(reader)
    tasks = {task["id"]: task for task in tasks}
    with open(CROSS_TASK_ROOT / "videos_val.csv") as f:
        val_samples = f.readlines()
    val_videos = set([x.split(",")[1] for x in val_samples])
    path_list = list(Path(CROSS_TASK_ROOT / "annotations").glob("*.csv"))
    print(f"cross has {len(path_list)} original samples")
    for path in path_list:
        try:
            splits = path.stem.split("_")
            task_id = splits[0]
            youtube_id = "_".join(splits[1:])
            sample_id = str2hashkey(path.name)
            if sample_id not in keys:
                keys.add(sample_id)
                with path.open() as f:
                    lines = [
                        line.strip() for line in f.readlines() if line.strip()
                    ]
                sample = {
                    "id": sample_id,
                    "youtube_id": youtube_id,
                    "ori": "cross",
                    "split": "test" if youtube_id in val_videos else "train",
                    "duration": -1,
                    "annotation": [
                        {
                            "clip_id": f"{sample_id}_{len(lines)}_{i}",
                            "segment": [float(x) for x in line.split(",")[1:3]],
                            "label": tasks[task_id]["steps"][
                                int(line.split(",")[0]) - 1
                            ],
                        }
                        for i, line in enumerate(lines)
                    ],
                }
                samples.append(sample)
            else:
                key_repeat += 1
        except Exception:
            print(path)
            print(tasks[task_id])
            raise
    print(f"cross has {len(samples)} samples")
    print(f"number of repeated keys: {key_repeat}")
    return samples


def var2vtt():

    with open(VAR_ROOT / "var_train_v1.0.json") as f:
        train = json.load(f)
    with open(VAR_ROOT / "var_val_v1.0.json") as f:
        val = json.load(f)
    with open(VAR_ROOT / "var_test_v1.0.json") as f:
        test = json.load(f)

    data = list(train.values()) + list(val.values()) + list(test.values())
    print(f"var has {len(data)} original samples")
    samples = []
    keys = set()
    key_repeat = 0
    for item in data:
        sample_id = str2hashkey(json.dumps(item["events"], indent=2), length=16)
        if sample_id not in keys:
            keys.add(sample_id)
            sample = {
                "id": sample_id,
                "youtube_id": item["events"][0]["video_id"],
                "ori": "var",
                "split": item["split"],
                "duration": item["events"][0]["duration"],
                "annotation": [
                    {
                        "clip_id": f"{sample_id}_{len(item['events'])}_{i}",
                        "segment": event["timestamp"],
                        "label": event["sentence"],
                    }
                    for i, event in enumerate(item["events"])
                ],
            }
            samples.append(sample)
        else:
            key_repeat += 1
    print(f"var has {len(samples)} samples")
    print(f"number of repeated keys: {key_repeat}")
    return samples


def coin2vtt():
    with jsonlines.open(COIN_ROOT / "data/videos.jsonl") as f:
        data = list(f)
    print(f"coin has {len(data)} original samples")
    samples = []
    keys = set()
    key_repeat = 0
    for item in data:
        sample_id = str2hashkey(json.dumps(item, indent=2), length=16)
        if sample_id not in keys:
            keys.add(sample_id)
            sample = {
                "id": sample_id,
                "youtube_id": item["id"],
                "ori": "coin",
                "split": item["subset"].replace("ing", ""),
                "duration": item["duration"],
                "annotation": [
                    {
                        "clip_id": f"{sample_id}_{len(item['annotation'])}_{i}",
                        "segment": step["segment"],
                        "label": step["label"],
                    }
                    for i, step in enumerate(item["annotation"])
                ],
            }
            samples.append(sample)
        else:
            key_repeat += 1
    print(f"coin has {len(samples)} samples")
    print(f"number of repeated keys: {key_repeat}")

    return samples


def preprocess():
    vtt_cross = cross2vtt()
    with jsonlines.open(VTT_ROOT / "cross.jsonl", mode="w") as writer:
        writer.write_all(vtt_cross)

    vtt_var = var2vtt()
    with jsonlines.open(VTT_ROOT / "var.jsonl", mode="w") as writer:
        writer.write_all(vtt_var)

    vtt_coin = coin2vtt()
    with jsonlines.open(VTT_ROOT / "coin.jsonl", mode="w") as writer:
        writer.write_all(vtt_coin)


def integrate():
    with jsonlines.open(VTT_ROOT / "cross.jsonl") as reader:
        vtt_cross = list(reader)
    with jsonlines.open(VTT_ROOT / "var.jsonl") as reader:
        vtt_var = list(reader)
    with jsonlines.open(VTT_ROOT / "coin.jsonl") as reader:
        vtt_coin = list(reader)
    vtt = vtt_cross + vtt_var + vtt_coin
    with jsonlines.open(VTT_ROOT / "vtt_all.jsonl", mode="w") as writer:
        writer.write_all(vtt)
    print(f"total samples: {len(vtt)}")

    # filter out samples with overlapping segments
    vtt = [
        sample for sample in vtt if not is_steps_overlap(sample["annotation"])
    ]
    overlap_samples = []
    non_overlap_samples = []
    for sample in vtt:
        if is_steps_overlap(sample["annotation"]):
            overlap_samples.append(sample)
        else:
            non_overlap_samples.append(sample)
    with jsonlines.open(VTT_ROOT / "vtt_overlap.jsonl", mode="w") as writer:
        writer.write_all(overlap_samples)
    with jsonlines.open(VTT_ROOT / "vtt_non_overlap.jsonl", mode="w") as writer:
        writer.write_all(non_overlap_samples)
    print(f"total samples after removing overlap: {len(non_overlap_samples)}")

    # filter out samples with too much steps
    MAX_STEPS = 12
    vtt = [
        sample
        for sample in non_overlap_samples
        if len(sample["annotation"]) <= MAX_STEPS
    ]
    with jsonlines.open(VTT_ROOT / "vtt.jsonl", mode="w") as writer:
        writer.write_all(vtt)
    print(
        f"total samples after removing steps greater than {MAX_STEPS}: {len(vtt)}"
    )


if __name__ == "__main__":
    preprocess()
    integrate()
