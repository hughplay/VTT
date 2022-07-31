import json
import pickle as pkl
from pathlib import Path
from typing import Callable

import jsonlines

LINE_CHANGE = "\n"


def read_lines(path, rm_empty_lines=False, strip=False):
    with open(path, "r") as f:
        lines = f.readlines()
    if strip:
        lines = [line.strip() for line in lines]
    if rm_empty_lines:
        lines = [line for line in lines if len(line.strip()) > 0]
    return lines


def write_lines(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        lines = [
            line if line.endswith(LINE_CHANGE) else f"{line}{LINE_CHANGE}"
            for line in lines
        ]
        f.writelines(lines)


def read_jsonlines(path):
    with jsonlines.open(path) as reader:
        samples = list(reader)
    return samples


def write_jsonlines(path, samples):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        writer.write_all(samples)


def list2dict(_list, key: str):
    return {item[key]: item for item in _list}


def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_text(path):
    with open(path, "r") as fd:
        data = fd.read()
    return data


def write_text(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fd:
        fd.write(text)


def read_pickle(path):
    with open(path, "rb") as f:
        data = pkl.load(f)
    return data


def write_pickle(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fd:
        pkl.dump(obj=obj, file=fd)


class JSONLList:
    def __init__(self, jsonl_path, filter_func: Callable = None):
        self.samples = read_jsonlines(jsonl_path)

        if filter is not None:
            self.samples = [
                sample for sample in self.samples if filter(filter_func, sample)
            ]
        self.sample_dict = {sample["id"]: sample for sample in self.samples}

    def save(self, path, force=False):
        if not force:
            assert not Path(path).is_file()
        write_jsonlines(path, self.samples)

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return self.samples.__iter__()

    def __contains__(self, idx):
        return idx in self.sample_dict

    def __getitem__(self, idx):
        if type(idx) == int:
            return self.samples[idx]
        elif type(idx) == str:
            return self.sample_dict[idx]
        elif type(idx) == slice:
            return self.samples[idx]
        else:
            print(idx, type(idx))
            raise TypeError
