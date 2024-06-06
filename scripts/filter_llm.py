import argparse
import sys

sys.path.append(".")  # noqa: E402
from src.utils.datatool import read_jsonlines, write_jsonlines  # noqa: E402


def extract_json(text):
    try:
        text = text.strip()
        if text.startswith("{") or text.startswith("```"):
            text = text.replace("```json", "")
            text = text.replace("```", "")
            text = eval(text)
            text = text["transformations"]
            for trans in text:
                assert type(trans) is str, f"{trans} is not str"
    except Exception as e:
        print(f"Error: {e}, text: {text}")
        return None
    return text


def filter_unwanted(f_in, f_out):
    data = read_jsonlines(f_in)
    filter_data = []
    for sample in data:
        if extract_json(sample["response"]) is not None:
            filter_data.append(sample)
    write_jsonlines(f_out, filter_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    filter_unwanted(args.input, args.output)
