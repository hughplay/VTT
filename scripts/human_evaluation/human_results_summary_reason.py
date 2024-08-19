import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.append(".")
import src.utils.datatool as dtool  # noqa: E402

RESULT_DIR = Path("/data/vtt/human_reason_evaluation")
# BAD_SAMPLE_IDX = [1336, 695, 23, 929]
BAD_SAMPLE_ID = []


# read results from raw data
files = RESULT_DIR.glob("*.json")
results = []
results_exp = defaultdict(list)
for file in files:
    with open(file, "r") as f:
        data = json.load(f)
    if data["id"] in BAD_SAMPLE_ID:
        continue
    data["fluency"] = int(data["fluency"])
    data["relevance"] = int(data["relevance"])
    data["logical_soundness"] = int(data["logical_soundness"])
    results_exp["human"].append(data)
    try:
        results.append(
            {
                "id": data["id"],
                "exp": "human",
                "test_id": int(file.stem.split("_")[-1]),
                "fluency": data["fluency"],
                "relevance": data["relevance"],
                "logical_soundness": data["logical_soundness"],
                "normal": data["normal"],
            }
        )
    except Exception:
        print(file)

# save results to separate exp files
SAVE_DIR = Path("docs/lists/human_results")
SAVE_DIR.mkdir(exist_ok=True, parents=True)
for exp, items in results_exp.items():
    dtool.write_jsonlines(
        SAVE_DIR / f"{exp}.jsonl", sorted(items, key=lambda x: x["id"])
    )
# show results with latex table
df = pd.DataFrame(results)
df = df.sort_values(by=["id", "exp"])

exps = df.exp.unique()
results_dict = {}
for exp in exps:
    results_dict[exp] = df[df.exp == exp]

results_table = {}
df_ttnet = df[df.exp == "ttnet"]


def significant(df_exp, key):
    _, p_value = stats.ttest_ind(df_ttnet[key], df_exp[key])
    return "$^\\dagger$" if p_value < 0.05 else ""


for exp in exps:
    df_exp = df[df.exp == exp]
    mean = df_exp[["fluency", "relevance", "logical_soundness"]].mean()
    std = df_exp[["fluency", "relevance", "logical_soundness"]].std()
    results_table[exp] = {
        "fluency": f"{mean['fluency']:.2f}",
        "relevance": f"{mean['relevance']:.2f}",
        "LS": f"{mean['logical_soundness']:.2f}",
    }
    # results_table[exp] = {
    #     "fluency": f"{mean['fluency']:.2f}{significant(df_exp, 'fluency')}",
    #     "relevance": f"{mean['relevance']:.2f}{significant(df_exp, 'relevance')}",
    #     "LS": f"{mean['logical_soundness']:.2f}{significant(df_exp, 'logical_soundness')}",
    # }

df_human = pd.DataFrame(results_table)[["human"]].T
print(df_human)
print()
print(df_human.style.to_latex(hrules=True))
