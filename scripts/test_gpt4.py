import base64
import os
import sys
from pathlib import Path

import dotenv
import httpx
import requests
from openai import OpenAI
from tqdm import tqdm

sys.path.append(".")  # noqa: E402
from src.utils.datatool import read_jsonlines, write_jsonlines  # noqa: E402


class GPT4Predictor:
    PROMPT = """There are {} panels in the picture of an event strip, and each panel shows one state of the event.
Write {} transformations between every two adjacent panels (states) to describe what happened to cause one state to transform into another.
Each transformation must be a phrase. Here are some examples from other pictures: "put steak on grill", "release liquid", "add whipped cream"...

Your answer must be strictly formatted as following JSON format, organized as a *single* string list as the value of the only key "transformations" of the dict:
{{
    "transformations": [{}]
}}
"""

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

    def __init__(self, api_key, model="gpt-4-turbo"):
        self.api_key = api_key
        self.client = self.init_client()
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def init_client(self):
        client = OpenAI(
            base_url="https://api.xty.app/v1",
            api_key=self.api_key,
            http_client=httpx.Client(
                base_url="https://api.xty.app/v1",
                follow_redirects=True,
            ),
        )
        return client

    def get_prompt(self, num_panels, multi=False):
        return self.PROMPT.format(
            num_panels, num_panels - 1, ", ".join(self.TEMPS[: num_panels - 1])
        )

    def predict(self, image_path):
        image = self.encode_image(image_path)
        num_panels = int(Path(image_path).stem.split("_")[-1])
        prompt = self.get_prompt(num_panels=num_panels)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 300,
        }
        response = requests.post(
            "https://api.xty.app/v1/chat/completions",
            headers=self.headers,
            json=payload,
        ).json()
        response_text = response["choices"][0]["message"]["content"]
        return response_text


def eval_gpt4(
    model="gpt-4-vision-preview",
    test_samples_path="scripts/human_evaluation/human_test_samples.jsonl",
    save_path="scripts/human_evaluation/human_test_samples_gpt4.jsonl",
    image_root="/data/reason/vtt/concat_states",
):
    dotenv.load_dotenv(override=True)
    api_key = os.getenv("API_KEY")
    predictor = GPT4Predictor(api_key, model=model)

    samples = read_jsonlines(test_samples_path)
    samples = [x for x in samples if x["split"] == "test"]

    save_path = Path(save_path)
    data_root = Path(image_root)
    saved_samples = []
    if not save_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        saved_samples = read_jsonlines(save_path)
    saved_samples_id = [sample["id"] for sample in saved_samples]
    for sample in tqdm(samples, ncols=80):
        # breakpoint()
        if sample["id"] in saved_samples_id:
            continue
        image_path = (
            data_root / f"{sample['id']}_{len(sample['annotation'])+1}.jpg"
        )
        try:
            response = predictor.predict(image_path=image_path)
            saved_samples.append(
                {
                    "id": sample["id"],
                    "response": response,
                }
            )
            write_jsonlines(save_path, saved_samples)
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    model = "gpt-4-vision-preview"
    # model = "gpt-4o"
    # test_samples_path = "scripts/human_evaluation/human_test_samples.jsonl"
    test_samples_path = "/data/reason/vtt/meta/vtt.jsonl"
    save_path = "/data/reason/vtt/llm_raw/vtt_test_samples_gpt4.jsonl"
    # save_path = "/data/reason/vtt/llm_raw/vtt_test_samples_gpt4o.jsonl"
    image_root = "/data/reason/vtt/concat_states"

    eval_gpt4(
        model=model,
        test_samples_path=test_samples_path,
        save_path=save_path,
        image_root=image_root,
    )
