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
- [ ] finetune image encoder with different learning rate
Discarded:
- [ ] ~~lmdb, run out of inodes, sad, too much small files~~
    - [ ] writing frames & into lmdb
    - [ ] lmdb dataloader
    - unless we need to use video frames after
- [ ] add topic category accuracy
- [ ] inference module, demo
- [ ] tune hyper parameters
    - optuna, check pytorch lightning & optuna document
    - major hyper parameters
        - learning rate
        - learning rate scheduler
        - batch size
        - dropout
        - *embedding dimension

## Currently Working

- [ ] text generation
    - [ ] beam search
    - [ ] min length
    - [ ] repeat word penealty
- [ ] demo for comparing results
- experiments:
    - [ ] difference features
    - [ ] use the idea of glocal features in ttnet
    - [ ] add loss functions
    - [x] compare scheduler
        - linear warmup v.s. constant warmup
        - start: 2022-08-27 00:03:06

## 2022-08-26 23:51:37

- [x] test difference image encoders
    - start: 2022-08-24 16:30:45
    - estimated end: 2022-08-25 afternoon
    - ViT-L/14@336px bug: RuntimeError: CUDA error: an illegal memory access was encountered
    - RN50x64 bug: RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input
    - consolusion: seems RN50x4 is the best
- [ ] decoder inference
    - sampling
        - [x] greedy
        - [ ] beam search
            - https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
        - [x] top_k top_p
        - how huggingface's generate works? https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/text_generation#generation
        - https://huggingface.co/blog/how-to-generate
        - integrate huggingface's generation utils
    - generate utils
        - key concepts
            - LogitProcessor: directly use `transformers.LogitsProcessor`s
                - purpose:
            - StoppingCriterion: directly use `transformers.StoppingCriterion`s
                - purpose:
        - give up using `transformers`, considering too many scenarios and too many configurations
        - major stages:
            1. prepare initial input_ids
            2. loop until max lengths reached
                1. prepare model inputs
                2. get next token logits
                3. adjust logits
                4. greedy or sample
- [x] enable BERTScore only during testing, too slow!
- [x] check overall models
    - [x] check label_ids offset
        - ~~cst, feature as start, exclude `<start-of-text>` in `label_ids`~~
        - give up deciding whether to prefix `<start-of-text>` or not
        - the right way is to tell the correct shift to generation loss
            - label_ids: <start-of-text> text <end-of-text>
            - label_mask: true ...true... true ...false...
            - cst
                - input: <feature> <start-of-text> text
                - output: <start-of-text> text <end-of-text>
                - logit_shift: -1
                    - output: text <end-of-text>
                - label_shift: -1
                    - label: text <end-of-text>
            - glacnet
                - input: <start_of_text> text
                - output: text <end-of-text>
                - logit_shift: 0
                    - output: text <end-of-text>
                - label_shift: -1
                    - label: text <end-of-text>
            - ttnet
                - input: <context> <start_of_text> text
                - ouptut: <start_of_text> text <end-of-text>
                - logit_shift: -1
                    - output: text <end-of-text>
                - label_shift: -1
                    - label: text <end-of-text>
- [x] test half-precision
    - runtime: -1h
    - performace:
        - cst: nan
        - glacnet: little drop
        - ttnet: almost same, some metrics drop, some metrics improve
    - conclusion: no half-precision for cst & glacnet, half-precision for ttnet
- [x] fix bug: max val/PPL -> min val/PPL
- [x] fix bug: ROUGE, METER computation, wrong average
= [x] issue: BERTScore is very slow when testing, seems to use CPU only
    - the default value of `device` is None, which means CPU
    - now we can also use BERTScore during validation
- [x] bug: generation config is under model, which is not correctly overriden during testing
    - pl use `update` to merge parameters into saved `Dictconfig`, which occurs error
    - use a temp file to save hparams and then load, according to `pl.load_from_checkpoint`
- [x] save results to files

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
