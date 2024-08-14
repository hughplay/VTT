import os
import sys
import time
from pathlib import Path

import dotenv
import google.ai.generativelanguage as glm
import google.generativeai as genai
from tqdm import tqdm

sys.path.append(".")  # noqa: E402
from src.utils.datatool import read_jsonlines, write_jsonlines  # noqa: E402

DEBUG = True


class GeminiPredictor:
    PRE_PROMPT = """There are {} panels in the picture of an event strip, and each panel shows one state of the event.
Your task is to first describe the event in the picture, and then I will show you every two adjacent panels (states)
one by one, you need to write the corresponding transformation between them, to describe what happened to cause one
state to transform into another based on your understanding of the whole event and the two given states. Each
transformation must be a phrase. Here are some examples from other pictures: "put steak on grill", "release liquid",
"add whipped cream"...

Describe the event in the picture:
"""
    TRANS_PROMPT = """Describe the No.{} transformation between No.{} and No.{} states with a phrase:"""

    def __init__(self, api_key, model="gemini-pro-vision"):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        # https://colab.research.google.com/drive/1kgr7HayLXCxMgh6IZl1PLw1WPuknMEBt?usp=sharing#scrollTo=RtTQqi1HC8O7
        self.history = []

    def upload_to_gemini(self, path, mime_type=None):
        """Uploads a file to Gemini for use in prompts."""
        file = genai.upload_file(path, mime_type=mime_type)
        if DEBUG:
            print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file

    def generate(self):
        num_retry = 0
        while True:
            try:
                response = self.model.generate_content(self.history)
                text = response.text.strip().strip("**")

                if text:
                    break

                if DEBUG:
                    print("Empty response, retrying...")

                num_retry += 1
                if num_retry > 3:
                    if DEBUG:
                        print("Max retry reached, returning empty string")
                    break

            except Exception as e:
                if DEBUG:
                    print(f"Error: {e}")
                    print("Retrying...")
                time.sleep(1)
        if DEBUG:
            print(text)

        self.history.append(response.candidates[0].content)
        return text

    def predict(self, images_path):
        num_panels = len(images_path)

        files = [
            self.upload_to_gemini(image_path)
            for image_path in tqdm(
                images_path, ncols=80, desc="Uploading files", leave=False
            )
        ]

        response = {}
        self.history.append(
            {
                "role": "user",
                "parts": [self.PRE_PROMPT.format(num_panels)] + files,
            }
        )
        response["overall"] = self.generate()

        response["preds"] = []
        for i in range(num_panels - 1):
            self.history.append(
                {
                    "role": "user",
                    "parts": [self.TRANS_PROMPT.format(i + 1, i + 1, i + 2)]
                    + files[i : i + 1],
                }
            )
            response["preds"].append(self.generate())

        return response


def eval_gemini(
    model,
    test_samples_path,
    save_path,
    image_root,
):
    dotenv.load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_API_KEY")
    predictor = GeminiPredictor(api_key, model=model)

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
        if sample["id"] in saved_samples_id:
            continue

        images_path = list(
            data_root.glob(f"{sample['id']}_{len(sample['annotation'])+1}*.jpg")
        )
        try:
            response = predictor.predict(images_path=images_path)
            saved_samples.append(
                {
                    "id": sample["id"],
                    "response": response,
                }
            )
            write_jsonlines(save_path, saved_samples)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
        finally:
            time.sleep(5)


if __name__ == "__main__":
    # model = "gemini-pro-vision"
    model = "gemini-1.5-pro-latest"
    test_samples_path = "/data/reason/vtt/meta/human_test_samples.jsonl"
    # save_path = "/data/reason/vtt/llm_raw/vtt_test_samples_gemini_multi.jsonl"
    save_path = "/data/reason/vtt/llm_multiturn/vtt_test_samples_gemini1.5_multiturn.jsonl"
    image_root = "/data/reason/vtt/states"

    eval_gemini(
        model=model,
        test_samples_path=test_samples_path,
        save_path=save_path,
        image_root=image_root,
    )
