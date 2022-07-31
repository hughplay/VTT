## TODO

- [ ] ?Retrain tokenizer

## Currently Working

- [ ] compare vist and vtt samples
- methods research, three major directions
  - visual storytelling
  - Open-domain Dialogue Generation
  - dense video captioning
- [ ] implement a baseline model
    - [ ] design baseline models

## 2022-08-01 01:20:02

- [x] dataloader
    - two modes
        - with intermediate video frames
        - without intermediate video frames
    - [x] tokenizer
        - **clip: bpe (SimpleTokenizer)**
            - max words: 21 + 2 (start, end tokens), set to 24
        - gpt2: bpe
            - the tokenized results from clip and gpt3 are different. oh no!
            - gpt2 has the same bos and eos token
        - var: nltk,
        - see MAGIC
            - MAGIC use GPT2 tokenizer
            - MAGIC add a `<-start_of_text->` to the tokenizer
        - bpe: a good introduction video: https://huggingface.co/course/chapter6/2
    - [x] dataset code, test
    - [x] Classify CrossTask tasks into COIN's categories
    - [x] Plot categories distribution
    - [x] train val test splitting
    - [x] test datamodule

add keys to samples:

``` json
{
    "id": "<sample_id>",
    "youtube_id": "<youtube_id>",
    "ori": "<original_dataset>",
    "split": "train/val/test",
    "duration": <duration>,
    "topic": "<topic of the video>",
    "category": "<category of the video>",
    "annotation": [
        {
            "clip_id": "<sample_id>_<total_clips>_<clip_idx>",
            "segment": [<start>, <end>],
            "label": "<description>",
        },
        ...
    ],
}
```


## 2022-7-27 16:52:39

Remove VAR from data. The reasons include:
- Sentences in VAR are more likely descriptions of states, rather than transformations or actions.
- Sentences are too long in VAR.
- Very few videos after filtering videos contains overlapping temporal boundaries.

- [x] add plotting fonts to base docker image
- [x] downloading VIST val split, for analysis of VIST samples, the question is:
how differently between "states" telling and "transformation" telling?
- define some max length
    - max transformations: 12
    - max states: 13
    - max words: 21 (19 from coin + bos + eos)

## 2022-7-22 16:28:06

Preprocessing pipeline:

1. pre_merge_dataset.py
2. pre_extract_states.py
3. pre_extract_clips.py
4. pre_extract_frames.py


## 2022-7-13 15:27:23

- [x] extract frames from videos

```py
clip_path = (
    Path(args.output)
    / f"{sample['id']}_{len(steps)}_{i}.mp4"
)
out_dir = Path(args.output) / f"{clip_path.stem}"
```

## 2022-07-09

- [x] extract states from videos

```py
t_states = [steps[0]["segment"][0]] + [
    step["segment"][1] for step in steps
]
out_path = (
    Path(args.output)
    / f"{sample['id']}_{len(t_states)}_{i}.jpg"
)
```


## 2022-07-03 21:04:49


Start to integrate three datasets, including COIN, ActivityNet, and CrossTask. The format of data:

``` json
{
    "id": "<sample_id>",
    "youtube_id": "<youtube_id>",
    "ori": "<original_dataset>",
    "split": "train/val/test",
    "duration": <duration>,
    "annotation": [
        {
            "clip_id": "<sample_id>_<total_clips>_<clip_idx>",
            "segment": [<start>, <end>],
            "label": "<description>",
        },
        ...
    ],
}
```
