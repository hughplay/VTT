## TODO

- proposals:
  - [ ] add stage position embedding
- [ ] ?Retrain tokenizer
- [ ] compare vist and vtt samples
- Visualize heatmap
    - pytorch-grad-cam: https://github.com/jacobgil/pytorch-grad-cam
    - torch-cam: https://github.com/frgfm/torch-cam
    - https://github.com/ml-tooling/best-of-ml-python#model-interpretability
        - https://github.com/slundberg/shap#deep-learning-example-with-gradientexplainer-tensorflowkeraspytorch-models
    - [x] GradCAM paper reading, 2022-08-19
- ffcv looks awesome
    - https://github.com/libffcv/ffcv
    - see this from: https://towardsdatascience.com/pytorch-lightning-vs-deepspeed-vs-fsdp-vs-ffcv-vs-e0d6b2a95719
- [ ] optimization for variable length transformations
- [ ] finetune image encoder with different learning rate
Discarded:
- [ ] ~~lmdb, run out of inodes, sad, too much small files~~
    - [ ] writing frames & into lmdb
    - [ ] lmdb dataloader
    - unless we need to use video frames after

## Currently Working

- [ ] decoder inference
    - sampling
        - greedy
        - beam search
            - https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
        - top_k top_p
        - how huggingface's generate works? https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/text_generation#generation

## 2022-08-22 17:46:48

- [x] uniform common used variables:
    - `states`: original images
    - `states_mask`: to mark varied number of states, (B, N)
    - `features`: encoded image features
    - `context`: contextualized image features
    - `end_context`: contextualized image features, but ignore the initial state
    - `trans_mask`: to mark varied number of transformation descriptions, `states_mask[:, 1:]` or `label_mask.sum(-1) > 0`
    - `label_ids`: captions
    - `label_mask`: mask for captions
- [x] loss functions
- [x] configurations
- [x] training logic
- [x] bug: torch metrics act differently when training and pytest
    - pytest: not sync dist
    - training: sync dist
    - fixed by hacking with states
- [x] make BERTScore not need to connect huggingface.co
    - `scripts/download_transformers.py`
    - `test/test_metrics.py`
    - `conf/criterion/telling_v1.py`
- [ ] check overall models
    - [x] check masking, by checking dataset
    - [x] check empty strings
        - ignore both (preds and target) empty strings in criterion
        - smart decode, skip start token, stop when end token is hit, return `""` if no end token
- [x] bug: dataset split not works


## 2022-08-15 14:26:45

- [x] dataloader: arguments to decide whether add <start> and <end>
- baseline models
    - [x] GLACNet
    - [x] CST (Contextualize, Show, and Tell)
    - [x] TTNet naive (ours)
    - visualize models
      - torchviz: https://github.com/szagoruyko/pytorchviz
      - hiddenlayer: https://github.com/waleedka/hiddenlayer/blob/master/demos/pytorch_graph.ipynb
      - [x] torch.onnx.export + netron
        - scripts/visualize_model.py
        - https://github.com/lutzroeder/netron
    - [x] testing

## 2022-08-13 15:48:01

- [x] decoder forward
    - LSTM decoder
        - Contextualize
        - glocal
    - Transformer
    - unit testing
- [x] break model into parts
    - image encoder API
    - context encoder
    - text decoder

## 2022-08-12 10:43:47

- [x] context encoder API
    - biLSTM, pack_padded_sequence
        - question is how biLSTM handle padded sequences
        - https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        - https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
    - [x] simple_lstm (Contextualize, Show, and Tell)
    - [x] glocal
    - [x] transformer
        - add classical fixed positional embedding to x_transformers, for future comparison
        - all position embeddings
            - fixed
            - absolute
            - infused fixed
            - simplified relative (T5)
    - [x] testing

## 2022-08-10 10:21:21

- [x] image encoder API
    - resnet
    - inception_v3
    - clip models
    - dim agnostic encode
- [x] test all models on cpu gpu

## 2022-08-07 13:53:06

- [x] methods research, three major directions
  - visual storytelling
  - Open-domain Dialogue Generation
  - dense video captioning
- [x] design baseline models
    - encoder:
        - CNN + Bi-LSTM/GRU
        - Graph Network
        - KB enriched, ConceptNet
    - decoder
        - BART
        - position embedding: the decoder should know the position of transformation in the whole story, near start or near end
   - loss
        - label smoothing loss
    - length difference position embedding (KG-Story)
    - generation
        - Repetition Penalty (KG-Story)
- [x] metrics
    - VIST:
      - METEOR
      - smoothed-BLEU (Lin and Och, 2004)
      - Skip-Thoughts (Kiros et al., 2015)
    - final list, read these papers
        - [x] METEOR
        - [x] BLUE-4: n-gram matching, modified
        - [x] CIDEr: TF-IDF as vector, cosine similarity
        - [x] ROUGE_L
        - [x] BERTScore: word embedding similarity, full pair-wise
    - question: transformation-wise or sample-wise evaluation?
        - transformation-wise for now



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
