import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(".")
import src.utils.datatool as dtool  # noqa: E402

RESULT_DIR = Path("/data/vtt/human_reason")
# BAD_SAMPLE_IDX = [1336, 695, 23, 929]
BAD_SAMPLE_ID = []


# read results from raw data
files = RESULT_DIR.glob("*.json")
results = []
for file in files:
    with open(file, "r") as f:
        data = json.load(f)
    if data["id"] in BAD_SAMPLE_ID:
        continue
    try:
        results.append(
            {
                "id": data["id"],
                "preds": data["preds"],
                "order": int(file.stem.split("_")[-1]),
            }
        )
    except Exception:
        print(file)

# save results to separate exp files
SAVE_PATH = Path("docs/lists/human_reason/human_reason.jsonl")
SAVE_PATH.parent.mkdir(exist_ok=True, parents=True)
dtool.write_jsonlines(SAVE_PATH, sorted(results, key=lambda x: x["order"]))
