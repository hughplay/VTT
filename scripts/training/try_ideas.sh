# Base CST shared
python train.py experiment=baseline_cst_shared

# Base TTNet using GLocal context features
python train.py experiment=baseline_ttnet_glocal

# *Base TTNet using LSTM text decoder
python train.py experiment=baseline_ttnet_lstm \
    criterion.loss.logit_shift=0 criterion.loss.label_shift=-1

# Tie Embedding
python train.py experiment=baseline_ttnet \
    model.tie_embedding=true name=baseline_ttnet_tie \
    callbacks.checkpoint.monitor="val/BLEU_4"

# *Fusion context into label ids by adding context to word embedding
python train.py experiment=baseline_ttnet_context \
    model.decoder_context_fusion=add name=baseline_ttnet_context_add

# *Fusion context into label ids by concatenation and linear projection
python train.py experiment=baseline_ttnet_context \
    model.decoder_context_fusion=concat name=baseline_ttnet_context_concat

# *Replace image encoder of CST with resnet152
python train.py experiment=baseline_cst \
    model.image_encoder="resnet152" name="baseline_cst_resnet152" \
    dataset.transform_cfg.train.transform_mode="imagenet"

# BiContext add
python train.py experiment=ttnet_bicontext \
    model.decoder_context_fusion=add name=baseline_ttnet_bicontext_add \
    logging.wandb.tags="[bicontext]"

# BiContext concat
python train.py experiment=ttnet_bicontext \
    model.decoder_context_fusion=concat name=baseline_ttnet_bicontext_concat \
    logging.wandb.tags="[bicontext]"

# Multitask BiContext add
python train.py experiment=ttnet_multitask \
    model.decoder_context_fusion=add model.bicontext=true \
    name=baseline_ttnet_multitask_bi_add \
    logging.wandb.tags="[bicontext]"

# Multitask BiContext concat
python train.py experiment=ttnet_multitask \
    model.decoder_context_fusion=concat model.bicontext=true \
    name=baseline_ttnet_multitask_bi_concat \
    logging.wandb.tags="[bicontext]"

# Classify BiContext add
python train.py experiment=ttnet_multitask_classify \
    model.decoder_context_fusion=add model.bicontext=true \
    name=baseline_ttnet_classify_bi_add \
    logging.wandb.tags="[bicontext]"

# Classify BiContext concat
python train.py experiment=ttnet_multitask_classify \
    model.decoder_context_fusion=concat model.bicontext=true \
    name=baseline_ttnet_classify_bi_concat \
    logging.wandb.tags="[bicontext]"

# MTM
python train.py experiment=ttnet_mtm

# MTM + diff new
python train.py experiment=sota_v4_diff \
    model.diff_mode=late \
    model.mask_ratio=0.15 \
    model.zero_prob=0.8 \
    model.random_prob=0.1 \
    model.topic_head=false \
    criterion.loss.w_classify=null \
    criterion.loss.w_topic=null \
    criterion.topic=false \
    name=ttnet_sota_v4_w_mtm2_no_topic_w_diff \
    logging.wandb.tags="[sota_v4]"

# densecap normalize weight
python train.py experiment=baseline_densecap \
    model.normalize_weight=true \
    name="baseline_densecap_norm" \
    logging.wandb.tags="[final_base]"
