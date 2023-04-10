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
    - Transformer visualization: https://openaccess.thecvf.com/content/CVPR2021/papers/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf
- [ ] finetune image encoder with different learning rate
- [ ] add topic category accuracy
- [ ] tune hyper parameters
    - optuna, check pytorch lightning & optuna document
    - major hyper parameters
        - learning rate
        - learning rate scheduler
        - batch size
        - dropout
        - *embedding dimension
- [ ] generation demo, features:
    - select models
    - select sample
    - select generation arguments
- [ ] text generation
    - [ ] beam search
- [ ] overall refinement?


## Currently Working

- [ ] design a structure constrained model for VTT
    - [x] provide definition
    - [ ] design models
        - [ ] image encoder decoder
            - [x] stable diffusion
            - [ ] dalle2 (dalle2-laion)
        - [ ] text encoder decoder
            - [ ] decap:
            - [ ] preprocess data
              -  WikiHow: https://github.com/mahnazkoupaee/WikiHow-Dataset
              - [x] split dataset, filter out tokens greater than 77
              - [x] wikihow text to clip embedding
                - lmdb-embeddings is very convenient
            - issue: testing results seems to be not good, test on coin descriptions
              - [ ] add a coin text dataset, and added it to the test data module


## 2022-11-19 23:55:32

- [x] mask ratio ablation
    - start: 2022-11-09 10:26:39
    - expected end: 2022-11-09 22:00:00
    - [x] mask table
- [x] sample mask ablation
    - start: 2022-11-09 22:59:23
    - expected end: 2022-11-10 10:30:00
- [x] add SPICE
    - [SPICE: Semantic Propositional Image Caption Evaluation](https://arxiv.org/pdf/1607.08822.pdf)
    - [ ] silent the print from SPICE
- [x] human evaluation specification
    - [Best practices for the human evaluation of automatically generated text](https://aclanthology.org/W19-8643.pdf)
- [x] human evaluation program
- [x] cross task annotation
    - Make Bicerin, oueddsUmOHI


## 2022.09.29

Time Table:

- 09.20 24:00
    - [x] check key components results
    - [x] check image encoder results
- 09.21 morning
    - [x] check difference results

- experiments:
    - [ ] add two baseline from dense video captioning
        - [x] DCE: "Dense-captioning events in videos"
            - FuturePastContext
            - ContextLSTM (from GLACNet)
        - [ ] MT: End-to-End Dense Video Captioning with Masked Transformer
            - https://arxiv.org/pdf/1804.00819v1.pdf
        - [ ] Bidirectional Attentive Fusion with Context Gating for Dense Video Captioning
            - https://arxiv.org/pdf/1804.00100v2.pdf
        - [ ] PDVC:
        - AI.M3: https://arxiv.org/pdf/2006.07896.pdf
        - BMT, MDVC, iPerceive, TSP, SDVC
    - [x] key components
    - [x] image encoder
    - [x] classification ablation
        - wclass_1_wcat_0_wtopic_0.25
        - wclass_1_wcat_0.0625_wtopic_0
    - [x] difference ablation
        - early difference
        - both difference
        - early difference only
        - late difference only
- experiment analysis
    - [ ] how difference features help
        - attention heatmap
    - [ ] how topic classification helps
        - case study
    - [ ] how MTM helps
        - case study

## 2022-09-20 10:28:01

Time Table:

- 09.19 21:00
    - [x] check diff + classify experiment results
- 09.20 10:00
    - [x] check MTM all random results
    - [x] check full experiment results*2

TODO:
- experiments:
    - [x] integrate:
        - final: ttnet_sota_v5_0.15_0.5_zero_wclass_0.25_wcat_0.1
        1. [x] ttnet base, no additional tricks
        2. [x] + ttnet diff
            - [x] late
            - [x] early
            - [x] early + late
            - no big difference, late is the best
        3. [x] diff + mtm
            - [x] 15% all zeros
            - [x] 15% learned mask
            - [x] MLM like strategies
            - [x] different mask ratio
                - [x] 5%
                - [x] 10%
                - [x] 15%
                - [x] 20%
                - [x] 25%
            - [x] different sample mask prob
                - [x] 100%
                - [x] 50%
                - [x] 25%
                - [x] 75%
            - the improvement is very small, hard to decide
            - [x] 25% all random
                - not better than all zero
        4. [x] diff + classify
        5. [x] diff + mtm + classify
            - select mtm strategy
                - 0.15 0.25 zero
                - *0.15 0.5 zero
                - 0.25 all zero
                - *0.2 all zero
                - *0.15 all zero
                - *0.1 all zero
            - select classify strategy
                - wclass 0.125 wcat 0.1
                - wclass 0.25 wcat 0.25
                - *wclass 0.25 wcat 0.1
                - *wclass 0.125 wcat 0.5
                - wclass 0.125 wcat 0.25
                - wclass 0.25 wcat 0.75
    - [x] adjust detailed mask ratio
        - detail:
            - 10%: unchanged
            - 80%: zero
            - 10%: random
        - start: 2022-09-15 16:19:13
        - not better than before
    - [ ] trade-off between multiple objectives
        - [Multi-Task Learning as Multi-Objective Optimization](https://proceedings.neurips.cc/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Paper.pdf)
        - [x] adjust weights of topic classification
            - 2
            - 4
            - 10
            - 100
            - 0.5
            - 0.25
            - 0.1
            - 0.01


## 2022-09-15 10:58:01

- research
    - difference features
        - key words: fine grained image classification, detect small visual changes
        - [Awesome Fine-Grained Image Analysis – Papers, Codes and Datasets](http://www.weixiushen.com/project/Awesome_FGIA/Awesome_FGIA.html)
        - MetaFormer : A Unified Meta Framework for Fine-Grained Recognition
    - text generation
        - keywords: multimodal text generation, text generation survey, context constrained (conditional) text generation
        - [ ] text generation based on existing text generation model
            - VL-ADAPTER: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks
                - https://arxiv.org/pdf/2112.06825.pdf
                - https://github.com/ylsung/VL_adapter
        - sampling: [Controllable Neural Text Generation](https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/)
        - [Generalized Visual Language Models](https://lilianweng.github.io/posts/2022-06-09-vlm/)
        - longformer
        - position embedding
    - multi-turn dialogue
        - key words: ReCoSa
- [x] move category, topic linear back to model
- questions:
    - evaluation metrics: sentence wise or sample wise?
- cases
    - difference might works:
        - 1277
    - mtm, topic should work
        - 1140
        - 42, breathing
    - repetition
        - 1142
- experiments:
    - [ ] overall refinement?
    - [x] fix: the target of the previous reconstruction is wrong (features), should be end_context
        - start: 2022-09-13 19:30:35
        - worse than wrong reconstruction = =
    - [x] difference features
        - how to extract:
            - [x] early difference features
            - [x] *late difference features
            - all pairs
        - how to contextualize:
            - simple attention
                - [x] diff first
                - [x] *diff last
            - [x] cross attention, diff as query
            - [x] concat and then linear project
        - start: 2022-09-13 14:25:43
        - conclusion:
            1. difference features imporve the perfomance with a large margin
            2. simple attention the most effective strategy
            3. late fusion is better than early fusion
            4. diff last is better than diff first
    - [x] what about use both early and late difference features?
        - start: 2022-09-14
        - not better than late difference features
    - [ ] category, topic classification
        - [Multi-Task Learning as Multi-Objective Optimization](https://proceedings.neurips.cc/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Paper.pdf)
    - [ ] decoder use cross attention
        - ReCoSa: Detecting the Relevant Contexts with Self-Attention for Multi-turn Dialogue Generation
    - [ ] attend and tell
        - context features with attention
    - [ ] transformer text decoder with cross attention
    - what we are sure:
        - ViT-L/14 is the best image encoder
        - on ViT-B/16, predicting topic, category, mtm all have positive effects
    - [x] sotav2
        - start: 2022-09-08
        - when topic classification is enabled, not mtm is better
        - when mtm is enabled, no classification is better
        - waiting for no mtm and no classification
        - large model makes large difference
    - [x] masked prediction
        - motivation: force model to generate descriptions based on images from other positions
        - start: 2022-09-08 10:31:37
        - ~+0.01
        - conclusion: masked prediction is helpful
    - [x] dropout for topic, category classification
        - motivation: overfitting
        - start: 2022-09-07 21:50:11
        - consolusion: negative effect
    - [x] currently sota ablation: sota_v1
        - the best
            - context + word embedding
            - multitask: category classification + topic classification
            - warmup: 500
        - ablation 1: image encoder
        - ablation 2: classification ablation
            - w/ topic is better than w/ topic, category
        - 2022-09-06
    - [x] add classification loss
        - start: 2022-09-02
        - CIDEr:
        - topic classification is the key
    - [x] add reconstruction loss
        - 2022-09-02
        - CIDEr: 4.259 -> 4.212
        - reconstruction has negative effect
    - [ ] stage position embedding
        - In [UNITER](https://github.com/ChenRocks/UNITER/blob/1dbfa62bb8ddb1f2b11d20775ce324c5e45a3ab4/model/model.py#L268), they project position features into a embedding vector with a linear layer.
    - [x] effect of training epochs
        - start: 2022-09-01
        - 100 epochs: 4.259 -> 4.417
    - [x] effect of warmup steps
        - start: 2022-09-02 00:30:18
        - affects little: 500 is the best, 4.259 (2000) -> 4.315 (500)
    - [x] why cst not good?
        - change image encoder to resnet152
            - start: 2022-09-01 19:08:09
            - CIDEr: 0.05 -> 0.5261
            - conclusion: something is wrong with inception v3
    - [x] bicontext
        - motivation: transformations are not consistent, must enchance the overall context information when decoding descriptions
        - start: 2022-09-04 20:14:56
        - little effect, almost no change
    - [x] bicontext multitask, classify
        - start: 2022-09-04 20:14:56
        - negative effect: -0.5 ~ -0.4 on CIDEr
    - image encoder
        - timm
            - https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv
        - [x] vision transformer
            - start: 2022-09-05 13:44:29
        - [x] swin transformer
            - start: 2022-09-05 13:44:29
        - [x] beit transformer
            - start: 2022-09-05 13:44:29
        - conclusion: CLIP models are better
    - [x] tie embedding
        - [Using the Output Embedding to Improve Language Models](https://arxiv.org/pdf/1608.05859v3.pdf)
        - [Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling](https://arxiv.org/pdf/1611.01462v3.pdf)
        - start: 2022-08-30 23:47:09
        - val/CIDEr is None and the training stopped
        - change the implementation from x_transformer's `x@embed.weight.t()` to `linear.weight=embed.weight`
            - no good luck
            - [ ] check the implemetation of BART, T5, GPT2

## 2022-09-01 13:35:19

- [x] bug: ppl not updated, directly use `perplexity.compute()`
- experiments:
    - [x] rerun experiments with updated settings:
            1. best model monitor: val/CIDEr
            2. generate from scratch during validation
            3. ttnet precision changed back to 32
        - start: 2022-08-29 19:29:21
        - conclusion: slightly better than previous experiments, ~+0.01
    - [ ] why cst not good?
        - [x] shared LSTM of cst?
            - start: 2022-08-30 11:09:49
            - BLEU_4 + 0.01
            - not because of shared LSTM
        - different image normalization?
            - imagenet's mean std is different with
            - test torchvision models with imagenet normalization
            - [x] rerun torchvision image encoders
                - start: 2022-08-30 10:19:34
                - conclusion: ViT-L/14 > ViT-B/16 > RN101 > ViT-B/32 > RN50x4 > resnet152 > inception_v3
            - glacnet +0.01, cst CIDEr+0.01 BERTScore +0.04
            - the effect of normalization is small
    - [x] why ttnet performs worse than cst?
        - [x] glocal feautres?
            - start: 2022-08-30 20:49:39
            - ~+0.01, not the key problem
        - [x] lstm decoder?
            - start: 2022-08-30 21:23:24
            - **performs better than glacnet!!!**
        - modify ttnet
            - [x] concat context with word embedding?
                - add context to word embedding
                    - start: 2022-08-31 11:27:57
                    - **yes! a large imporvement: CIDEr: 0.96 -> 4.25 > 3.50 (glacnet)**
                    - concat has similar performance
            - [ ] ~~use the idea of glocal features in ttnet~~
                - canceled because the glocal is not the key

Discarded:
- [ ] ~~lmdb, run out of inodes, sad, too much small files~~
    - [ ] writing frames & into lmdb
    - [ ] lmdb dataloader
    - unless we need to use video frames after
- ffcv looks awesome
    - https://github.com/libffcv/ffcv
    - see this from: https://towardsdatascience.com/pytorch-lightning-vs-deepspeed-vs-fsdp-vs-ffcv-vs-e0d6b2a95719

## 2022-08-29 17:10:33

- experiments:
    - [x] compare scheduler
        - linear warmup v.s. constant warmup
        - start: 2022-08-27 00:03:06
        - conclusion: constant warmup is slightly better
- [x] text generation
    - [x] min length
    - [x] repeatition word penealty
        - Conditional Transformer Language Model for Controllable Generation
            - https://arxiv.org/pdf/1909.05858.pdf
            - https://github.com/salesforce/ctrl/blob/master/generation.py
        - vocab length: 49408, end_idx: 49097
        - the order of logit processors is important
            - top_k top_p should be the last two processors
        - A Theoretical Analysis of the Repetition Problem in Text Generation, AAAI
            - high inflow problem
            - https://arxiv.org/pdf/2012.14660.pdf
            - https://github.com/fuzihaofzh/repetition-problem-nlg
        - directly ban last word seems to crash the prediction
- [x] demo for comparing results
    - https://nicjac.dev/posts/how-to-build-machine-learning-demo-in-2022/
    - inference: TorchServe
    - [x] Gradio
    - Streamlit
- [x] bug: BLEU not appears in the results
    - use pycocoeval BLEU, it returs 4 scores, use the last one which should be BLEU@4
    - https://leimao.github.io/blog/BLEU-Score/
    - torchmetrics BLEU will return 0 if n-gram denominator is 0, which is not desired
    - using smooth strategy: http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf
    - final method: nltk bleu score + smooth.method7
- [x] bug?: BERTScore is very large even though the results looks bad
    - the results between our results and `Tiiiger/bert_score` are different
    - rescale with baseline is needed!
        - https://github.com/Tiiiger/bert_score/blob/master/journal/rescale_baseline.md#rescaling-bertscore-with-baselines
- [x] CIDEr bug: use wrong variable to save offsets
- val PPL is too small to be used to monitor the performance of the checkpoint
    - [x] change the monitor metric from PPL to CIDEr
        - it seems all metrics are highly correlated with human judgement
    - [x] change val to generate from scratch
- TTNet performs very bad
    - change back to precision 32

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
