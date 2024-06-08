import os
from tqdm import tqdm
import jsonlines
from PIL import Image
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)


def load_jsonl(fname):
    datas = []
    with open(fname, "r") as f:
        for item in jsonlines.Reader(f):
            datas.append(item)
            
    return datas

TEMPS = [
    "<the 1st transformation you wrote>",
    "<the 2nd transformation you wrote>",
    "<the 3rd transformation you wrote>",
    "<the 4st transformation you wrote>",
    "<the 5st transformation you wrote>",
    "<the 6st transformation you wrote>",
    "<the 7st transformation you wrote>",
    "<the 8st transformation you wrote>",
    "<the 9st transformation you wrote>",
    "<the 10st transformation you wrote>",
    "<the 11st transformation you wrote>",
    "<the 12st transformation you wrote>",
    "<the 13st transformation you wrote>",
    ]

PROMPT = '''USER:
There are {} panels in the picture of an event strip, each showing one state of the event.
Write {} transformations between every two adjacent panels to describe what happened between two states that caused a state change.
Each transformation must be a phrase. Here are some transformations from other pictures: "insert oil gun in the car", "release liquid", "put steak on grill", "absorb liquid with dropper", "add whipped cream"...

Your answer must be formatted as JSON:
{{
"Transformations": [{}]
}}

ASSISTANT:
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test qwen on vtt")
    parser.add_argument('--data_path', type=str, default="/data/reason/vtt/meta/vtt.jsonl")
    parser.add_argument('--image_dir',type=str, default="/data/reason/vtt/concat_states")
    parser.add_argument('--output_path',type=str, default="./res/qwen-vl-chat_concat_0shot_output_v2.jsonl")
    parser.add_argument('--device',type=str, default="cuda:0")

    args = parser.parse_args()

    # format of data
    '''
    {"id": "e7cc4aadf3070857", "youtube_id": "NZ_5TD0l_WA", "ori": "coin", "split": "train", "duration": 89.42, "topic": "Fuel Car", "category": "Vehicle", "annotation": [{"clip_id": "e7cc4aadf3070857_4_0", "segment": [0.0, 7.0], "label": "open the fuel tank cap"}, {"clip_id": "e7cc4aadf3070857_4_1", "segment": [8.0, 21.0], "label": "insert oil gun in the car"}, {"clip_id": "e7cc4aadf3070857_4_2", "segment": [60.0, 65.0], "label": "pullthe  oil gun out"}, {"clip_id": "e7cc4aadf3070857_4_3", "segment": [66.0, 71.0], "label": "close the fuel tank cap"}], "frames": {"e7cc4aadf3070857_4_0": {"frames": 210, "imgs": 210}, "e7cc4aadf3070857_4_1": {"frames": 390, "imgs": 390}, "e7cc4aadf3070857_4_2": {"frames": 150, "imgs": 150}, "e7cc4aadf3070857_4_3": {"frames": 150, "imgs": 150}}}
    '''

    # load all data
    vtt_data = load_jsonl(args.data_path)
    # filter test data
    test_data = {d["id"]: d for d in vtt_data if d["split"]=="test"}

    # get all image files
    img_files = os.listdir(args.image_dir)
    # filter test image
    img_files = [i for i in img_files if i.split("_")[0] in test_data]

    assert len(img_files) == len(test_data), "number of data and image inconsistant"

    os.makedirs("/".join(args.output_path.split("/")[:-1]), exist_ok=True)

    already_done = set()
    if os.path.exists(args.output_path):
        res = load_jsonl(args.output_path)
        for r in res:
            already_done.add(r["id"])

    # load model
    tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("./pretrained_models/Qwen-VL-Chat", device_map=args.device, trust_remote_code=True).eval()


    with jsonlines.open(args.output_path, "a") as f:
        for n, img_file in tqdm(enumerate(img_files), total=len(img_files), desc="testing"):
            qid = img_file.split("_")[0]

            # filter already done
            if qid in already_done:
                continue
            
            event_num = int(img_file.split("_")[1].split(".")[0])

            query = tokenizer.from_list_format([
                {'image': os.path.join(args.image_dir, img_file)},
                {'text': PROMPT.format(event_num, event_num-1, ", ".join(TEMPS[:event_num-1]))},
            ])

            with torch.no_grad():
                inputs = tokenizer(query, return_tensors='pt')
                inputs = inputs.to(model.device)
                pred = model.generate(**inputs)
                response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)

                try:
                    i = response.index("ASSISTANT:")
                    response = response[i+len("ASSISTANT:"):].strip()
                    i = response.index("\"Transformations\":")
                    response = response[i+len("\"Transformations\":"):].replace("[", "").replace("]", "").replace("}", "").strip()
                    response = [r.strip() for r in response.split(",")]
                except:
                    pass
            res = {
                "id": qid,
                "response": response
                }
            f.write(res)

            if n < 3:
                print(PROMPT.format(event_num, event_num-1, ", ".join(TEMPS[:event_num-1])))
                print(response)
                print("="*10)
                



