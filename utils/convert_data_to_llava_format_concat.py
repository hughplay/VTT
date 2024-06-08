import os
from tqdm import tqdm
import jsonlines
import json
import argparse

def load_jsonl(fname):
    datas = []
    with open(fname, "r") as f:
        for item in jsonlines.Reader(f):
            datas.append(item)
            
    return datas


# PROMPT = '''<image>
# Write {} transformations between every two adjacent panels of an event strip.
# The transformation describes what happened between two states that caused a state change.
# '''

# PROMPT = '''{}
# Write {} transformations between every two adjacent panels of an event strip.
# The transformation describes what happened between two states that caused a state change.
# '''

PROMPT = '''{}
Write the topic of this event and {} transformations between every two adjacent panels of an event strip.
The transformation describes what happened between two states that caused a state change.
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert vtt format to llava")
    parser.add_argument('--data_path', type=str, default="/data/reason/vtt/meta/vtt.jsonl")
    parser.add_argument('--image_dir',type=str, default="/data/reason/vtt/concat_states")
    parser.add_argument('--output_dir',type=str, default="./data/llava_topic")

    args = parser.parse_args()

    # format of data
    '''
    {
        "id": "e7cc4aadf3070857", 
        "youtube_id": "NZ_5TD0l_WA", 
        "ori": "coin", 
        "split": "train", 
        "duration": 89.42, 
        "topic": "Fuel Car", 
        "category": "Vehicle", 
        "annotation": [{"clip_id": "e7cc4aadf3070857_4_0", "segment": [0.0, 7.0], "label": "open the fuel tank cap"}, {"clip_id": "e7cc4aadf3070857_4_1", "segment": [8.0, 21.0], "label": "insert oil gun in the car"}, {"clip_id": "e7cc4aadf3070857_4_2", "segment": [60.0, 65.0], "label": "pullthe  oil gun out"}, {"clip_id": "e7cc4aadf3070857_4_3", "segment": [66.0, 71.0], "label": "close the fuel tank cap"}], 
        "frames": {"e7cc4aadf3070857_4_0": {"frames": 210, "imgs": 210}, "e7cc4aadf3070857_4_1": {"frames": 390, "imgs": 390}, "e7cc4aadf3070857_4_2": {"frames": 150, "imgs": 150}, "e7cc4aadf3070857_4_3": {"frames": 150, "imgs": 150}}}
    '''
    # get all image files
    img_files = os.listdir(args.image_dir)

    # load all data
    vtt_data = load_jsonl(args.data_path)

    # assert len(img_files) == len(vtt_data), f"number of data ({len(vtt_data)}) and image ({len(img_files)}) inconsistant"

    # split data
    train_data = {d["id"]: d for d in vtt_data if d["split"]=="train"}
    val_data = {d["id"]: d for d in vtt_data if d["split"]=="val"}
    test_data = {d["id"]: d for d in vtt_data if d["split"]=="test"}

    all_data = {
        "train": [],
        "val": [],
        "test": [],
    }
    for img_file in tqdm(img_files, desc="convert dataset format"):
        qid = img_file.split("_")[0]
        event_num = int(img_file.split("_")[1].split(".")[0])
        if qid in train_data:
            split = "train"
            data = train_data[qid]
        elif qid in val_data:
            split = "val"
            data = val_data[qid]
        elif qid in test_data:
            split = "test"
            data = test_data[qid]
        else:
            continue

        annotation = "\n".join(["{}. {}".format(i, a["label"]) for i,a in enumerate(data["annotation"])])
        images = ["{}_{}_{}.jpg".format(qid, event_num, i) for i in range(event_num)]
        # annotation = "Topic: {}\nTransformations:\n".format(data["topic"])
        # annotation += "\n".join(["{}. {}".format(i, a["label"]) for i,a in enumerate(data["annotation"])])

        data = {
            "id": qid,
            "images": images,
            "conversations": [
                {
                    "from": "human",
                    "value": PROMPT.format("<image>"*event_num, event_num-1),
                },
                {
                    "from": "gpt",
                    "value": annotation
                },
            ]
        }

        all_data[split].append(data)

    os.makedirs(args.output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        with open(os.path.join(args.output_dir, f"{split}.json"), "w") as f:
            json.dump(all_data[split], f)


