import os
from tqdm import tqdm
import jsonlines
import json
import torch
from PIL import Image
import argparse
import re

import sys 
sys.path.append('../LLaVA')

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from transformers.generation.streamers import TextIteratorStreamer
import shutil

import requests
from io import BytesIO


class Predictor():
    def setup(self, model_path, model_name, model_base, device="cuda"):
        self.device = device
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            model_name=model_name, 
            model_base=model_base, 
            load_8bit=False, 
            load_4bit=False,
            device_map=device
            )
        self.model.eval()
    
    def predict(
        self,
        image=None,
        prompt=None,
        top_p=1.0,
        temperature=0.2,
        max_tokens=1024,
    ):
        """Run a single prediction on the model"""
    
        conv_mode = "v1"
        conv = conv_templates[conv_mode].copy()
    
        image_data = load_image(str(image))
        image_tensor = self.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().to(self.device)
    
        # loop start
    
        # just one turn, always prepend image token
        # inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        inp = prompt
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    
        with torch.no_grad():
            res = self.model.generate(
                inputs=input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens
            )
            res = self.tokenizer.decode(res[0], skip_special_tokens=True).strip()
            return res

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_jsonl(fname):
    datas = []
    with open(fname, "r") as f:
        for item in jsonlines.Reader(f):
            datas.append(item)
            
    return datas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test llava on vtt")
    parser.add_argument('--data_path', type=str, default="data/llava_concat_topic/test.json")
    parser.add_argument('--image_dir',type=str, default="/data/reason/vtt/concat_states")
    parser.add_argument('--output_path',type=str, default="./res/llava-v15-7b_concat_topic_lora_{}epoch_output.jsonl")
    parser.add_argument('--model_path',type=str, default="/data/reason/vtt/checkpoints/llava-v1.5-7b-lora-vtt-concat-topic-50epoch")
    parser.add_argument('--model_name',type=str, default="llava-v1.5-7b-lora-vtt-concat-topic")
    parser.add_argument('--model_base',type=str, default="./pretrained_models/liuhaotian/llava-v1.5-7b")
    parser.add_argument('--device',type=str, default="cuda")
    parser.add_argument('--start',type=int, default=0)
    parser.add_argument('--end',type=int, default=50)

    args = parser.parse_args()

    # format of data
    '''
    {"id": "6a00c03b88aa2bb9", "image": "6a00c03b88aa2bb9_6.jpg", "conversations": [{"from": "human", "value": "<image>\nWrite the topic of this event and 5 transformations between every two adjacent panels of an event strip.\nThe transformation describes what happened between two states that caused a state change.\n"}, {"from": "gpt", "value": "Topic: Make Meringue\nTransformations:\n0. pour egg\n1. whisk mixture\n2. add sugar\n3. put mixture into bag\n4. spread mixture"}]}
    '''

    # load all data
    with open(args.data_path, "r") as f:
        test_data = json.load(f)

    os.makedirs("/".join(args.output_path.split("/")[:-1]), exist_ok=True)

    checkpoints = [d for d in os.listdir(args.model_path) if d.startswith("checkpoint-")]
    checkpoints = [int(d.split("-")[-1]) for d in checkpoints]
    e = 5
    for i, checkpoint in enumerate(checkpoints):
        if i+1+e < args.start:
            continue
        if i+1+e >= args.end:
            break
        model_path = os.path.join(args.model_path, f"checkpoint-{checkpoint}")
        output_path = args.output_path.format(e+i+1)

        for fname in ["config.json", "non_lora_trainables.bin"]:
            src = os.path.join(args.model_path, fname)
            dst = os.path.join(model_path, fname)
            shutil.copy2(src, dst)

        already_done = set()
        if os.path.exists(output_path):
            res = load_jsonl(output_path)
            for r in res:
                already_done.add(r["id"])

        # load model
        predictor = Predictor()
        predictor.setup(model_path, args.model_name, args.model_base, args.device)

        with jsonlines.open(output_path, "a") as f:
            for d in tqdm(test_data, total=len(test_data), desc=f"testing ckpt {e+i+1}"):
                qid = d["id"]

                # filter already done
                if qid in already_done:
                    continue
                
                trans_num = int(d["image"].split("_")[1].split(".")[0])-1
                prompt = d["conversations"][0]["value"]

                response_ = predictor.predict(
                    image=os.path.join(args.image_dir, d["image"]),
                    prompt=prompt,
                    temperature=0.1
                    )
                i = response_.index("Transformations:")
                response_ = response_[i+16:].strip()

                response_ = response_.split("\n")
                response = []
                for i in range(trans_num):
                    if i >= len(response_):
                        response.append("none")
                    else:
                        r = re.sub(r'\d+\.\s*', '', response_[i]).strip()
                        response.append(r)
                res = {
                    "id": qid,
                    "response": response
                    }
                f.write(res)

                



