import argparse
import sys
from pathlib import Path

import netron
import torch

sys.path.append(".")
from src.model.components.image_encoder import _INPUT_OUTPUT_DIM  # noqa: E402
from src.model.components.text_decoder import N_WORDS  # noqa: E402
from src.model.cst import CST  # noqa: E402
from src.model.glacnet import GLACNet  # noqa: E402
from src.model.ttnet import TTNet  # noqa: E402
from src.utils.systool import human_readable_size  # noqa: E402


def export_onnx(model_name, export_file, train_mode):

    if model_name == "cst":
        model = CST()
    elif model_name == "glacnet":
        model = GLACNet()
    elif model_name == "ttnet":
        model = TTNet()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # prepare inputs
    batch_size = 1
    img_dim = _INPUT_OUTPUT_DIM[model.image_encoder.name]["input"]
    max_transformations = 12
    max_states = max_transformations + 1
    max_words = 24
    # inputs: images
    images = torch.randn(batch_size, max_states, 3, img_dim, img_dim)
    states_mask = torch.zeros(batch_size, max_states).bool()
    rand_len = torch.randint(1, max_states, (batch_size,))
    for i in range(batch_size):
        states_mask[i, : rand_len[i]] = True
    # inputs: captions
    captions = torch.randint(
        0, N_WORDS, (batch_size, max_transformations, max_words)
    )
    captions_mask = torch.zeros(
        batch_size, max_transformations, max_words
    ).bool()
    rand_len = torch.randint(1, max_words, (batch_size, max_states))
    for i in range(batch_size):
        for j in range(max_transformations):
            captions_mask[i, j, : rand_len[i, j]] = True

    torch.onnx.export(
        model,
        (images, states_mask, captions, captions_mask),
        export_file,
        verbose=True,
        export_params=False,
        training=torch.onnx.TrainingMode.TRAINING
        if train_mode
        else torch.onnx.TrainingMode.EVAL,
        input_names=["images", "states_mask", "captions", "captions_mask"],
        output_names=["logits"],
        opset_version=12,
    )


def visualize(model_name, export_dir, recreate, ip, port, train_mode):

    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    export_file = export_dir / f"{model_name}.onnx"

    if not export_file.exists() or recreate:
        print("Exporting model to ONNX..., this may take a while")
        export_onnx(model_name, export_file, train_mode)
        print(f"ONNX has been exported to {export_file}")
    else:
        print(
            f"Visualize model from existing file: {export_file}, "
            "use '--recreate' to recreate."
        )
    print(
        f"Start to visualize model {model_name}: "
        f"{human_readable_size(export_file.stat().st_size)}"
    )
    print("It may take a while to load the model in the browser, take a break.")

    netron.start(str(export_file), address=(ip, port), log=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="ttnet")
    parser.add_argument("-d", "--export_dir", type=str, default="/log/onnx")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--train-mode", action="store_true")

    args = parser.parse_args()

    visualize(
        args.model,
        args.export_dir,
        args.recreate,
        args.ip,
        args.port,
        args.train_mode,
    )
