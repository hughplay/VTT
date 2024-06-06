import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(".")
import src.utils.datatool as dtool  # noqa: E402

CURRENT_DIR = Path(__file__).parent
META_FILE = Path("/data/vtt/meta/vtt.jsonl")
EXPERIMENTS = {
    "gemini1.5": "/data/vtt/llm_result/vtt_test_samples_gemini1.5_multi.jsonl",
    "llava": "/data/vtt/llm_result/llava-v15-7b_concat_topic_0shot_output.jsonl",
    "llava_lora": "/data/vtt/llm_result/llava-v15-7b_concat_topic_lora_39epoch_output.jsonl",
}


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


def generate_result_list(samples):
    selected_results = []
    for exp_name, result_path in EXPERIMENTS.items():
        results = dtool.read_jsonlines(result_path)
        results_dict = {x["id"]: x for x in results}
        metrics_path = result_path.replace(".jsonl", "_detail.jsonl")
        metrics = dtool.read_jsonlines(metrics_path)
        metrics_dict = {x["id"]: x for x in metrics}
        for sample in samples:
            sample_id = sample["id"]
            labels = [x["label"] for x in sample["annotation"]]
            result = {
                "id": sample_id,
                "preds": results_dict[sample_id]["transformations"],
                "label": labels,
                "exp": exp_name,
            }
            for metric, score_list in metrics_dict[sample_id][
                "metrics"
            ].items():
                result[metric] = score_list
            selected_results.append(result)
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
    all_samples = dtool.read_jsonlines(CURRENT_DIR / "human_test_samples.jsonl")

    result_list = generate_result_list(all_samples)
    print("Result list length:", len(result_list))
    result_list = mark_perfect_results(result_list)
    random.shuffle(result_list)
    dtool.write_jsonlines(
        CURRENT_DIR / "human_test_results_llm.jsonl", result_list
    )


if __name__ == "__main__":
    main()
