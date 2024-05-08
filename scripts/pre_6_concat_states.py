from pathlib import Path

from sh import montage
from tqdm import tqdm


def concat_states(paths: list, output_path: str, padding: int = 2):
    output = montage(
        "-mode",
        "concatenate",
        "-geometry",
        f"+{padding}+{padding*2}",
        "-tile",
        "x1",
        *paths,
        output_path,
    )
    return output


def prepare_states(root_path: str, output_root_path: str):
    Path(output_root_path).mkdir(parents=True, exist_ok=True)
    states = {}
    for path in tqdm(
        Path(root_path).glob("*.jpg"), ncols=80, desc="Prepare states"
    ):
        key = "_".join(path.stem.split("_")[:-1])
        if key not in states:
            states[key] = {
                "paths": [],
                "output_path": Path(output_root_path) / f"{key}.jpg",
            }
        states[key]["paths"].append(path)
    return states


def main():
    root_path = "/data/reason/vtt/states"
    output_root_path = "/data/reason/vtt/concat_states"
    states = prepare_states(root_path, output_root_path)
    for value in tqdm(states.values(), ncols=80, desc="Concat states"):
        concat_states(value["paths"], value["output_path"])


if __name__ == "__main__":
    main()
