import random
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger

sys.path.append(".")
import src.utils.datatool as dtool  # noqa: E402

DATA_ROOT = Path("/data")
VTT_ROOT = DATA_ROOT / "vtt" / "meta"


def integrate_information():
    samples = dtool.read_jsonlines(VTT_ROOT / "vtt_pre.jsonl")
    logger.info(f"total samples: {len(samples)}")

    state_info = dtool.list2dict(
        dtool.read_jsonlines(VTT_ROOT / "report_extract_states.jsonl"), key="id"
    )
    clip_info = dtool.list2dict(
        dtool.read_jsonlines(VTT_ROOT / "report_extract_clips.jsonl"), key="id"
    )
    frame_info = dtool.list2dict(
        dtool.read_jsonlines(VTT_ROOT / "report_extract_frames.jsonl"), key="id"
    )

    filter_samples = []
    for sample in samples:
        _id = sample["id"]

        if _id not in state_info:
            logger.warning(f"{_id} not in state_info")
        elif _id not in clip_info:
            logger.warning(f"{_id} not in clip_info")
        elif _id not in frame_info:
            logger.warning(f"{_id} not in frame_info")

        # only reserve successfully preprocessed samples
        elif state_info[_id]["status"] != "success":
            logger.warning(
                f"{_id} extracted states failed, "
                f"status is {state_info[_id]['status']}"
            )
        elif clip_info[_id]["status"] != "success":
            logger.warning(
                f"{_id} extracted clips failed, "
                f"status is {clip_info[_id]['status']}"
            )
        elif frame_info[_id]["status"] != "success":
            logger.warning(
                f"{_id} extracted frames failed, "
                f"status is {frame_info[_id]['status']}"
            )
        else:
            sample["duration"] = state_info[_id]["duration"]
            sample["frames"] = frame_info[_id]["frames"]
            for clip_id, f in sample["frames"].items():
                if f["imgs"] == 0:
                    logger.warning(f"{clip_id} has no extracted frames")
                if f["frames"] != f["imgs"]:
                    logger.warning(
                        f"{clip_id} has {len(sample['annotation'])} steps"
                    )
            filter_samples.append(sample)

    logger.info(f"samples after filtering: {len(filter_samples)}")
    dtool.write_jsonlines(VTT_ROOT / "vtt_integrated.jsonl", filter_samples)
    return samples


def split_samples(samples):
    """split samples train:val:test = 8:1:1 at topic granularity"""
    random.seed(0)
    topic_samples = defaultdict(list)
    for sample in samples:
        topic_samples[sample["topic"]].append(sample)
    for s in topic_samples.values():
        random.shuffle(s)
        for sample in s[: int(len(s) * 0.8)]:
            sample["split"] = "train"
        for sample in s[int(len(s) * 0.8) : int(len(s) * 0.9)]:
            sample["split"] = "val"
        for sample in s[int(len(s) * 0.9) :]:
            sample["split"] = "test"
    random.shuffle(samples)

    # statistics
    train_samples = [s for s in samples if s["split"] == "train"]
    val_samples = [s for s in samples if s["split"] == "val"]
    test_samples = [s for s in samples if s["split"] == "test"]
    logger.info(f"train: {len(train_samples)}")
    logger.info(f"val: {len(val_samples)}")
    logger.info(f"test: {len(test_samples)}")

    dtool.write_jsonlines(VTT_ROOT / "vtt.jsonl", samples)
    return samples


if __name__ == "__main__":
    samples = integrate_information()
    split_samples(samples)
