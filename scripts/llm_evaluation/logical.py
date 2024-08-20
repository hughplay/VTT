# ! pip install -q -U google-generativeai

import os
import time

import google.generativeai as genai
import jsonlines
from tqdm import tqdm

genai.configure(api_key="AIzaSyBdT-4P_atF5t22Jm1ZXLHa-2P0ga8gJdw")

model = genai.GenerativeModel("gemini-1.5-pro")

prompt = """Impartially assign a score for the transformation sequence ranging from 1 to 5. \
A transformation sequence corresponds to an event, \
where each transformation describes the change between two adjacent states in the event.
Each transformation in a sequence is separated by a comma.
Your scoring needs to be only considered from the perspective of logical consistency. \
Ignore other aspects, such as grammar, spelling, fluency, vividness, etc.
The meaning of each score is as follows:
5: The logic between the transformation descriptions is consistent with commonsense.
4: The logic between most of the descriptions is consistent with commonsense.
3: The logic between some of the descriptions is consistent with commonsense.
2: There seems to be logic between the descriptions, but it doesn't make commonsense.
1: There is no logic between the transformation descriptions, or they are completely inconsistent with commonsense.

transformation sequence: {}
your score (output a numerical score directly without any extra explanation):"""


for name in [
    "gemini1.5",
    "densecap",
    "ttnet_base",
    "glacnet",
    "llava_lora",
    "llava",
    "cst",
    "ttnet",
]:
    input_file = "/content/{}.jsonl".format(name)
    output_file = "/content/llm_eval_v2/{}.jsonl".format(name)
    res = {}

    def load_jsonl(fname):
        datas = []
        with open(fname, "r") as f:
            for item in jsonlines.Reader(f):
                datas.append(item)

        return datas

    samples = load_jsonl(input_file)

    already_id = set()
    if os.path.exists(output_file):
        res = load_jsonl(output_file)
        for r in res:
            already_id.add(r["id"])

    with jsonlines.open(output_file, "a") as f:
        for sample in tqdm(samples):
            qid = sample["id"] if "id" in sample else sample["index"]
            if qid in already_id:
                # print(f"skip {qid}")
                continue

            preds = ", ".join(sample["preds"]).lower()

            # response = model.generate_content(
            #     prompt.format(preds),
            #     generation_config = genai.GenerationConfig(
            #         max_output_tokens=4,
            #     ))
            # s = response.text
            # r = {
            #     "id": qid,
            #     "logical_soundness": s,
            # }
            # f.write(r)
            # already_id.add(qid)

            response = model.generate_content(
                prompt.format(preds),
                generation_config=genai.GenerationConfig(
                    max_output_tokens=4,
                ),
            )
            s = response.text
            r = {
                "id": qid,
                "logical_soundness": int(s.strip()),
            }
            f.write(r)
            already_id.add(qid)
