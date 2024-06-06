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


class GeminiPredictor:
    PROMPT = """There are {} panels in the picture of an event strip, and each panel shows one state of the event.
Write {} transformations between every two adjacent panels (states) to describe what happened to cause one state to transform into another.
Each transformation must be a phrase. Here are some examples from other pictures: "put steak on grill", "release liquid", "add whipped cream"...

Your answer must be formatted as JSON:
{{
    "transformations": [{}]
}}
"""
    MULTI_PROMPT = """There are {} pictures of an event strip, and each picture shows one state of the event.
Write {} transformations between every two adjacent pictures (states) to describe what happened to cause one state to transform into another.
Each transformation must be a phrase. Here are some examples from other pictures: "put steak on grill", "release liquid", "add whipped cream"...

Your answer must be formatted as JSON:
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

    def __init__(self, api_key, model="gemini-pro-vision"):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def get_prompt(self, num_panels, multi=False):
        prompt = self.MULTI_PROMPT if multi else self.PROMPT
        return prompt.format(
            num_panels, num_panels - 1, ", ".join(self.TEMPS[: num_panels - 1])
        )

    def predict(self, image_path):
        num_panels = int(Path(image_path).stem.split("_")[-1])
        prompt = self.get_prompt(num_panels=num_panels)
        response = self.model.generate_content(
            glm.Content(
                parts=[
                    glm.Part(text=prompt),
                    glm.Part(
                        inline_data=glm.Blob(
                            mime_type="image/jpeg",
                            data=Path(image_path).read_bytes(),
                        )
                    ),
                ],
            ),
            stream=True,
        )
        response.resolve()
        return response.text.strip()

    def predict_images(self, images_path):
        num_panels = len(images_path)
        prompt = self.get_prompt(num_panels=num_panels, multi=True)
        response = self.model.generate_content(
            glm.Content(
                parts=[glm.Part(text=prompt)]
                + [
                    glm.Part(
                        inline_data=glm.Blob(
                            mime_type="image/jpeg",
                            data=Path(image_path).read_bytes(),
                        )
                    )
                    for image_path in images_path
                ]
            )
        )
        response.resolve()
        return response.text.strip()


def eval_gemini(
    model,
    test_samples_path,
    save_path,
    image_root,
    multi_image_root=None,
):
    dotenv.load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_API_KEY")
    predictor = GeminiPredictor(api_key, model=model)

    samples = read_jsonlines(test_samples_path)
    samples = [x for x in samples if x["split"] == "test"]

    save_path = Path(save_path)
    data_root = Path(image_root)
    multi_data_root = Path(multi_image_root)
    saved_samples = []
    if not save_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        saved_samples = read_jsonlines(save_path)
    saved_samples_id = [sample["id"] for sample in saved_samples]
    for sample in tqdm(samples, ncols=80):
        if sample["id"] in saved_samples_id:
            continue

        success = False
        # image_path = (
        #     data_root / f"{sample['id']}_{len(sample['annotation'])+1}.jpg"
        # )
        # try:
        #     response = predictor.predict(image_path=image_path)
        #     saved_samples.append(
        #         {
        #             "id": sample["id"],
        #             "response": response,
        #         }
        #     )
        #     write_jsonlines(save_path, saved_samples)
        #     if model == "gemini-1.5-pro-latest":  # rate limit
        #         time.sleep(30)
        #     success = True
        # except Exception as e:
        #     print(f"Error: {e}")

        if not success:
            images_path = list(
                multi_data_root.glob(
                    f"{sample['id']}_{len(sample['annotation'])+1}*.jpg"
                )
            )
            try:
                response = predictor.predict_images(images_path=images_path)
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
                if model == "gemini-1.5-pro-latest":  # rate limit
                    time.sleep(30)
                else:
                    time.sleep(1)


if __name__ == "__main__":
    # model = "gemini-pro-vision"
    model = "gemini-1.5-pro-latest"
    test_samples_path = "/data/reason/vtt/meta/vtt.jsonl"
    # save_path = "/data/reason/vtt/llm_raw/vtt_test_samples_gemini_multi.jsonl"
    save_path = (
        "/data/reason/vtt/llm_raw/vtt_test_samples_gemini1.5_multi.jsonl"
    )
    image_root = "/data/reason/vtt/concat_states"
    mult_image_root = "/data/reason/vtt/states"

    eval_gemini(
        model=model,
        test_samples_path=test_samples_path,
        save_path=save_path,
        image_root=image_root,
        multi_image_root=mult_image_root,
    )
