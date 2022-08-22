import argparse
from pathlib import Path

from transformers import AutoModel, AutoTokenizer


def save_model(model_name, model_path, proxy):

    # Do this on a machine with internet access
    model = AutoModel.from_pretrained(
        model_name, proxies={"http": proxy, "https": proxy}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    _ = model.save_pretrained(model_path)
    _ = tokenizer.save_pretrained(model_path)

    print(f"Files are saved to {model_path}:")
    for file in Path(model_path).iterdir():
        print(f"  - {file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="roberta-large")
    parser.add_argument(
        "--root", type=str, default="/data/pretrain/transformers"
    )
    parser.add_argument("--proxy", type=str)
    args = parser.parse_args()

    save_model(args.name, Path(args.root) / args.name, args.proxy)
