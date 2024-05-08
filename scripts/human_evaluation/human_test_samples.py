import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(".")
import src.utils.datatool as dtool  # noqa: E402

CURRENT_DIR = Path(__file__).parent
META_FILE = Path("/data/vtt/meta/vtt.jsonl")
EXPERIMENTS = {
    "cst": "/log/exp/vtt/VTTDataModule.CST.GenerationLoss.2022-09-17_00-06-25",
    "glacnet": "/log/exp/vtt/VTTDataModule.GLACNet.GenerationLoss.2022-09-18_17-56-36",
    "densecap": "/log/exp/vtt/VTTDataModule.DenseCap.GenerationLoss.2022-09-25_15-15-34",
    "ttnet_base": "/log/exp/vtt/VTTDataModule.TTNetMTM.GenerationLoss.2022-09-16_10-59-03",
    "ttnet": "/log/exp/vtt/VTTDataModule.TTNetDiff.TellingLossV1.2022-09-19_21-44-42",
}
EXTRA_SAMPLES_IDX = [412, 1359]


def random_sample(data, per_topic=1):
    topics_data = defaultdict(list)
    selected_samples_idx = []
    for sample in data:
        topics_data[sample["topic"]].append(sample)
    for topic, samples in topics_data.items():
        data_idx = [data.index(x) for x in random.sample(samples, per_topic)]
        selected_samples_idx.extend(data_idx)
        print(
            f"Topic {topic}: {len(samples)} samples. "
            f"Selected sample: {data_idx}"
        )
    return selected_samples_idx


def generate_result_list(samples_idx):
    selected_results = []
    for exp_name, exp_path in EXPERIMENTS.items():
        result_path = Path(exp_path) / "detail.jsonl"
        results = dtool.read_jsonlines(result_path)
        for sample_idx in samples_idx:
            results[sample_idx]["exp"] = exp_name
            results[sample_idx]["sample_idx"] = sample_idx
            selected_results.append(results[sample_idx])
    return selected_results


def mark_perfect_results(results):
    perfect_count = 0
    for result in results:
        label = [x.strip() for x in result["label"]]
        preds = [x.strip() for x in result["preds"]]
        result["perfect"] = label == preds
        if result["perfect"]:
            perfect_count += 1
    print(f"Perfect count: {perfect_count}")
    return results


def main():
    random.seed(2023)

    data = dtool.JSONLList(META_FILE, lambda x: x["split"] == "test").samples
    dtool.write_jsonlines(CURRENT_DIR / "vtt_test.jsonl", data)

    random_samples_idx = random_sample(data, per_topic=1)
    all_samples_idx = random_samples_idx + EXTRA_SAMPLES_IDX
    random.shuffle(all_samples_idx)
    all_samples = [data[idx] for idx in all_samples_idx]

    print(all_samples_idx)

    dtool.write_jsonlines(CURRENT_DIR / "human_test_samples.jsonl", all_samples)

    result_list = generate_result_list(all_samples_idx)
    print("Result list length:", len(result_list))
    result_list = mark_perfect_results(result_list)
    random.shuffle(result_list)
    dtool.write_jsonlines(CURRENT_DIR / "human_test_results.jsonl", result_list)


if __name__ == "__main__":
    main()
