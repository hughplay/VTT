import hashlib
import json
import re
from pathlib import Path

import jsonlines

DATA_ROOT = Path("/data")
CROSS_TASK_ROOT = DATA_ROOT / "CrossTask" / "crosstask_release"
VAR_ROOT = DATA_ROOT / "VAR"
COIN_ROOT = DATA_ROOT / "coin"
VTT_ROOT = DATA_ROOT / "vtt" / "meta"


def str2hashkey(s, length=16):
    return hashlib.md5(s.encode()).hexdigest()[:length]


def is_steps_overlap(steps):
    for i in range(len(steps) - 1):
        if steps[i]["segment"][1] > steps[i + 1]["segment"][0]:
            return True
    return False


""" Categories from COIN
00. Nursing and Care
01. Vehicle
02. Leisure and Performance
03. Gadgets
04. Electrical Appliance
05. Furniture and Decoration
06. Science and Craft
07. Pets and Fruit
08. Drink and Snack
09. Dish
10. Sport
11. Housework
"""

category_mapping = {
    "Make Jello Shots": "Drink and Snack",
    "Build Simple Floating Shelves": "Furniture and Decoration",
    "Make Taco Salad": "Drink and Snack",
    "Grill Steak": "Dish",
    "Make Kimchi Fried Rice": "Dish",
    "Make Meringue": "Drink and Snack",
    "Make a Latte": "Drink and Snack",
    "Make Bread and Butter Pickles": "Dish",
    "Make Lemonade": "Drink and Snack",
    "Make French Toast": "Dish",
    "Jack Up a Car": "Vehicle",
    "Make Kerala Fish Curry": "Dish",
    "Make Banana Ice Cream": "Drink and Snack",
    "Add Oil to Your Car": "Vehicle",
    "Change a Tire": "Vehicle",
    "Make Irish Coffee": "Drink and Snack",
    "Make French Strawberry Cake": "Dish",
    "Make Pancakes": "Dish",
    "Make a Black Forest Cake": "Dish",
    "Make Lavender Lemonade": "Drink and Snack",
    "Make Gummy Bears": "Drink and Snack",
    "Make Prawn Curry": "Dish",
    "Make Dutch Coffee": "Drink and Snack",
    "Make Papaya Salad": "Drink and Snack",
    "Cook Tuna Steak": "Dish",
    "Make Garlic Dill Pickles": "Dish",
    "Build a Desk": "Furniture and Decoration",
    "Make London Broil": "Dish",
    "Make Tiramisu Coffee": "Drink and Snack",
    "Make Vegan French Toast": "Dish",
    "Cook Italian Sausage": "Dish",
    "Build a Bookshelf": "Furniture and Decoration",
    "Cook Brazilian Rice": "Dish",
    "Make a Christmas Cake": "Dish",
    "Pickle Okra": "Dish",
    "Make Steak Teppanyaki": "Dish",
    "Make Masala Sauce": "Dish",
    "Grill Kabobs": "Dish",
    "Can Tomato Sauce": "Dish",
    "Change a Hubcap": "Vehicle",
    "Make Challah French Toast": "Dish",
    "Make Peppermint Meringue Cookies": "Drink and Snack",
    "Change Wheel Bearings": "Vehicle",
    "Make Eggnog French Toast": "Dish",
    "Make Sourdough Pancakes": "Dish",
    "Make Vanilla Custard": "Drink and Snack",
    "Make Real Vanilla Bean Ice Cream": "Drink and Snack",
    "Change The Brake Pads in Your Car": "Vehicle",
    "Make Limeade": "Drink and Snack",
    "Make Tuna Salad": "Drink and Snack",
    "Make Buttermilk Toaster Pancakes": "Dish",
    "Make Mocha Nutella Gelato": "Drink and Snack",
    "Grill Flank Steak": "Dish",
    "Make Tuticorin Macaroons": "Drink and Snack",
    "Make Chili Con Carne": "Dish",
    "Install Lowering Springs": "Vehicle",
    "Remove Lug Nuts and Tires": "Vehicle",
    "Make Stuffed Eggplant with Fish": "Dish",
    "Make Banana Pudding": "Drink and Snack",
    "Make Battenburg Cake": "Dish",
    "Grill Pork Tenderloin": "Dish",
    "Make a Grilled Cheese and Pickle Sandwich": "Dish",
    "Install a Fireplace Mantel": "Furniture and Decoration",
    "Make Tomato Rice": "Dish",
    "Grill Swordfish": "Dish",
    "Make Watermelon Lemonade": "Drink and Snack",
    "Make Potato Pancakes": "Dish",
    "Make Steamed Rice": "Dish",
    "Make Bicerin": "Drink and Snack",
    "Make a Caramel Macchiato": "Drink and Snack",
    "Make Edamame Corn Salad": "Drink and Snack",
    "Make Low Carb Coffee Cake": "Dish",
    "Make an Americano": "Drink and Snack",
    "Make Fish Stew": "Dish",
    "Make Rainbow Jello": "Drink and Snack",
    "Make Mad Eye Martini Jello Shots": "Drink and Snack",
    "Make Vegan Meringue": "Drink and Snack",
    "Make a Frappe": "Drink and Snack",
    "Grill Tri Tip": "Dish",
    "Make Chinese Fried Rice": "Dish",
    "Make Blueberry Pancakes": "Dish",
    "Make a Blended Iced Cappuccino": "Drink and Snack",
    "Make a Cafe Au Lait": "Drink and Snack",
}


class ListData:
    def __init__(self, data_list):
        self._data_list = data_list
        self._id_map = {sample["id"]: sample for sample in data_list}

    def __getitem__(self, _id):
        return self._id_map[_id]

    def __len__(self):
        return len(self._data_list)

    def __iter__(self):
        return self._data_list.__iter__()


class Taxonomy:
    def __init__(self, json_path=COIN_ROOT / "data" / "taxonomy.json"):
        with open(json_path) as f:
            self._data = json.load(f)
        self.domains = ListData(self._data["domain"])
        self.targets = ListData(self._data["target"])
        self.actions = ListData(self._data["action"])

    def get_domain_targets(self, domain_id):
        domain = self.domains[domain_id]
        targets = [self.targets[_id] for _id in domain["target_list"]]
        return targets

    def get_target_actions(self, target_id):
        target = self.targets[target_id]
        actions = [self.actions[_id] for _id in target["action_list"]]
        return actions

    def get_action_target(self, action_id):
        return self.targets[self.actions[action_id]["target_id"]]

    def get_target_domain(self, target_id):
        return self.domains[self.targets[target_id]["domain_id"]]

    def get_action_domain(self, action_id):
        target = self.get_action_target(action_id)
        return self.domains[target["domain_id"]]

    def split_words(self, s):
        # split words by Capital letter
        words = re.findall(r"CPR|RJ45|SIM|SSD|CD|TV|PC|[A-Z][^A-Z]*", s)
        words = " ".join(words)
        return words


def cross2vtt():
    samples = []
    keys = set()
    key_repeat = 0
    with jsonlines.open(CROSS_TASK_ROOT / "tasks.jsonl") as reader:
        tasks = list(reader)
    tasks = {task["id"]: task for task in tasks}
    with open(CROSS_TASK_ROOT / "videos_val.csv") as f:
        val_samples = f.readlines()
    val_videos = set([x.split(",")[1] for x in val_samples])
    path_list = list(Path(CROSS_TASK_ROOT / "annotations").glob("*.csv"))
    print(f"cross has {len(path_list)} original samples")
    for path in path_list:
        try:
            splits = path.stem.split("_")
            task_id = splits[0]
            youtube_id = "_".join(splits[1:])
            sample_id = str2hashkey(path.name)
            if sample_id not in keys:
                keys.add(sample_id)
                with path.open() as f:
                    lines = [
                        line.strip() for line in f.readlines() if line.strip()
                    ]
                sample = {
                    "id": sample_id,
                    "youtube_id": youtube_id,
                    "ori": "cross",
                    "split": "test" if youtube_id in val_videos else "train",
                    "duration": -1,
                    "topic": tasks[task_id]["task"],
                    "category": category_mapping[tasks[task_id]["task"]],
                    "annotation": [
                        {
                            "clip_id": f"{sample_id}_{len(lines)}_{i}",
                            "segment": [float(x) for x in line.split(",")[1:3]],
                            "label": tasks[task_id]["steps"][
                                int(line.split(",")[0]) - 1
                            ],
                        }
                        for i, line in enumerate(lines)
                    ],
                }
                samples.append(sample)
            else:
                key_repeat += 1
        except Exception:
            print(path)
            print(tasks[task_id])
            raise
    print(f"cross has {len(samples)} samples")
    print(f"number of repeated keys: {key_repeat}")
    return samples


def var2vtt():

    with open(VAR_ROOT / "var_train_v1.0.json") as f:
        train = json.load(f)
    with open(VAR_ROOT / "var_val_v1.0.json") as f:
        val = json.load(f)
    with open(VAR_ROOT / "var_test_v1.0.json") as f:
        test = json.load(f)

    data = list(train.values()) + list(val.values()) + list(test.values())
    print(f"var has {len(data)} original samples")
    samples = []
    keys = set()
    key_repeat = 0
    for item in data:
        sample_id = str2hashkey(json.dumps(item["events"], indent=2), length=16)
        if sample_id not in keys:
            keys.add(sample_id)
            sample = {
                "id": sample_id,
                "youtube_id": item["events"][0]["video_id"],
                "ori": "var",
                "split": item["split"],
                "duration": item["events"][0]["duration"],
                "annotation": [
                    {
                        "clip_id": f"{sample_id}_{len(item['events'])}_{i}",
                        "segment": event["timestamp"],
                        "label": event["sentence"],
                    }
                    for i, event in enumerate(item["events"])
                ],
            }
            samples.append(sample)
        else:
            key_repeat += 1
    print(f"var has {len(samples)} samples")
    print(f"number of repeated keys: {key_repeat}")
    return samples


def coin2vtt():
    with jsonlines.open(COIN_ROOT / "data/videos.jsonl") as f:
        data = list(f)
    print(f"coin has {len(data)} original samples")
    samples = []
    keys = set()
    key_repeat = 0
    taxonomy = Taxonomy(COIN_ROOT / "data" / "taxonomy.json")
    for item in data:
        sample_id = str2hashkey(json.dumps(item, indent=2), length=16)
        if sample_id not in keys:
            keys.add(sample_id)
            sample = {
                "id": sample_id,
                "youtube_id": item["id"],
                "ori": "coin",
                "split": item["subset"].replace("ing", ""),
                "duration": item["duration"],
                "topic": taxonomy.split_words(item["class"]),
                "category": taxonomy.get_target_domain(item["recipe_type"])[
                    "label"
                ],
                "annotation": [
                    {
                        "clip_id": f"{sample_id}_{len(item['annotation'])}_{i}",
                        "segment": step["segment"],
                        "label": step["label"],
                    }
                    for i, step in enumerate(item["annotation"])
                ],
            }
            samples.append(sample)
        else:
            key_repeat += 1
    print(f"coin has {len(samples)} samples")
    print(f"number of repeated keys: {key_repeat}")

    return samples


def preprocess():
    vtt_cross = cross2vtt()
    with jsonlines.open(VTT_ROOT / "cross.jsonl", mode="w") as writer:
        writer.write_all(vtt_cross)

    # vtt_var = var2vtt()
    # with jsonlines.open(VTT_ROOT / "var.jsonl", mode="w") as writer:
    #     writer.write_all(vtt_var)

    vtt_coin = coin2vtt()
    with jsonlines.open(VTT_ROOT / "coin.jsonl", mode="w") as writer:
        writer.write_all(vtt_coin)


def integrate():
    with jsonlines.open(VTT_ROOT / "cross.jsonl") as reader:
        vtt_cross = list(reader)
    with jsonlines.open(VTT_ROOT / "coin.jsonl") as reader:
        vtt_coin = list(reader)
    vtt = vtt_cross + vtt_coin
    with jsonlines.open(VTT_ROOT / "vtt_all.jsonl", mode="w") as writer:
        writer.write_all(vtt)
    print(f"total samples: {len(vtt)}")

    # filter out samples with overlapping segments
    vtt = [
        sample for sample in vtt if not is_steps_overlap(sample["annotation"])
    ]
    overlap_samples = []
    non_overlap_samples = []
    for sample in vtt:
        if is_steps_overlap(sample["annotation"]):
            overlap_samples.append(sample)
        else:
            non_overlap_samples.append(sample)
    with jsonlines.open(VTT_ROOT / "vtt_overlap.jsonl", mode="w") as writer:
        writer.write_all(overlap_samples)
    with jsonlines.open(VTT_ROOT / "vtt_non_overlap.jsonl", mode="w") as writer:
        writer.write_all(non_overlap_samples)
    print(f"total samples after removing overlap: {len(non_overlap_samples)}")

    # filter out samples with too much steps
    MAX_STEPS = 12
    vtt = [
        sample
        for sample in non_overlap_samples
        if len(sample["annotation"]) <= MAX_STEPS
    ]
    with jsonlines.open(VTT_ROOT / "vtt_pre.jsonl", mode="w") as writer:
        writer.write_all(vtt)
    print(
        f"total samples after removing steps greater than {MAX_STEPS}: {len(vtt)}"
    )


if __name__ == "__main__":
    VTT_ROOT.mkdir(exist_ok=True, parents=True)
    preprocess()
    integrate()
