import os
from tqdm import tqdm
import jsonlines
import argparse
import re

def load_jsonl(fname):
    datas = []
    with open(fname, "r") as f:
        for item in jsonlines.Reader(f):
            datas.append(item)
            
    return datas

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/data/reason/vtt/meta/vtt.jsonl")
    parser.add_argument('--input_path', type=str, default="res_old/qwen-vl-chat_split_0shot_output_v2.jsonl")
    parser.add_argument('--output_path',type=str, default="res/qwen-vl-chat_split_0shot_output.jsonl")

    args = parser.parse_args()

    # format of data
    '''
    {"id": "e7cc4aadf3070857", "youtube_id": "NZ_5TD0l_WA", "ori": "coin", "split": "train", "duration": 89.42, "topic": "Fuel Car", "category": "Vehicle", "annotation": [{"clip_id": "e7cc4aadf3070857_4_0", "segment": [0.0, 7.0], "label": "open the fuel tank cap"}, {"clip_id": "e7cc4aadf3070857_4_1", "segment": [8.0, 21.0], "label": "insert oil gun in the car"}, {"clip_id": "e7cc4aadf3070857_4_2", "segment": [60.0, 65.0], "label": "pullthe  oil gun out"}, {"clip_id": "e7cc4aadf3070857_4_3", "segment": [66.0, 71.0], "label": "close the fuel tank cap"}], "frames": {"e7cc4aadf3070857_4_0": {"frames": 210, "imgs": 210}, "e7cc4aadf3070857_4_1": {"frames": 390, "imgs": 390}, "e7cc4aadf3070857_4_2": {"frames": 150, "imgs": 150}, "e7cc4aadf3070857_4_3": {"frames": 150, "imgs": 150}}}
    '''
    vtt_data = load_jsonl(args.data_path)
    test_data = {d["id"]: d for d in vtt_data if d["split"]=="test"}

    # format of output
    '''
    {"id": "6a00c03b88aa2bb9", "response": "1. A person is holding a measuring cup.\n2. The measuring cup is being filled with a substance.\n3. The substance is being poured into a bowl.\n4. The bowl is being placed on a table.\n5. The person is holding a knife."}
    '''
    output_ = load_jsonl(args.input_path)

    with jsonlines.open(args.output_path, "w") as f:
        for o in output_:
            qid = o["id"]
            data = test_data[qid]
            event_num = len(data["annotation"])
            response_ = o["response"]
            response = []
            for i in range(event_num):
                if i >= len(response_):
                    response.append("none")
                else:
                    response.append(response_[i])

            res = {
                "id": qid,
                "response": response
                }
            f.write(res)
