import argparse
import sys
from pathlib import Path

import torch
from lmdb_embeddings.writer import LmdbEmbeddingsWriter
from tqdm import tqdm

sys.path.append(".")
from src.model.components.clip.clip import tokenize  # noqa: E402
from src.model.components.text_encoder import TextCLIP  # noqa: E402
from src.utils.datatool import read_lines  # noqa: E402

DATA_ROOT = Path("/data/vtt/wikihow")
BATCH_SIZE = 64
CUDA = True


def batch2clip(model, batch):
    batch = tokenize(batch)
    if CUDA:
        batch = batch.cuda()
    with torch.no_grad():
        batch = model(batch)
    batch = batch.cpu().numpy()
    return batch


def iter_embeddings(model_name):
    samples = {
        "train": read_lines(DATA_ROOT / "wikihow_train.txt", strip=True),
        "val": read_lines(DATA_ROOT / "wikihow_val.txt", strip=True),
        "test": read_lines(DATA_ROOT / "wikihow_test.txt", strip=True),
    }
    for split, split_samples in samples.items():
        print(f"{split}: {len(split_samples)}")

    text_encoder = TextCLIP(name=model_name, fixed=True)

    if CUDA:
        text_encoder = text_encoder.cuda()

    for split, split_samples in samples.items():
        batch = []
        pos_complete = 0
        for sample in tqdm(list(split_samples), ncols=80):
            batch.append(sample)
            if len(batch) == BATCH_SIZE:
                batch = batch2clip(text_encoder, batch)
                for i, sample in enumerate(batch):
                    yield f"{split}_{pos_complete + i}", sample
                pos_complete += len(batch)
                batch = []
        else:
            if batch:
                batch = batch2clip(text_encoder, batch)
                for i, sample in enumerate(batch):
                    yield f"{split}_{pos_complete + i}", sample


def main(model_name):
    lmdb_path = str(DATA_ROOT / f"wikihow_{model_name.replace('/', '_')}.lmdb")
    writer = LmdbEmbeddingsWriter(iter_embeddings(model_name)).write(lmdb_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
    )
    args = parser.parse_args()
    main(args.model_name)
