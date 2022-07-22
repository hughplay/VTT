## 2022-7-22 16:28:06

Preprocessing pipeline:

1. pre_merge_dataset.py
2. pre_extract_states.py
3. pre_extract_clips.py
4. pre_extract_frames.py


## 2022-7-13 15:27:23

- [x] extract frames from videos


## 2022-07-09

- [x] extract states from videos


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
