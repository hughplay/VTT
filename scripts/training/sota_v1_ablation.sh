# sota v1
python train.py experiment=sota_v1 \
    logging.wandb.tags="[sota_v1,multitask,baseline]"

# multitask ablation
# -topic
python train.py experiment=sota_v1 \
    criterion.loss.w_topic=0. \
    name=ttnet_sota_v1_no_topic \
    logging.wandb.tags="[sota_v1,multitask]"

# -category
python train.py experiment=sota_v1 \
    criterion.loss.w_category=0. \
    name=ttnet_sota_v1_no_category \
    logging.wandb.tags="[sota_v1,multitask]"

# -topic -category
python train.py experiment=sota_v1_no_classify \
    name=ttnet_sota_v1_no_classify \
    logging.wandb.tags="[sota_v1,multitask]"

# image encoder ablation
# resnet152
python train.py experiment=sota_v1 \
    model.image_encoder="resnet152" \
    name=ttnet_sota_v1_resnet152 \
    dataset.transform_cfg.train.transform_mode="imagenet" \
    logging.wandb.tags="[sota_v1,image_encoder]"

# beit_large_patch16_224
python train.py experiment=sota_v1 \
    model.image_encoder="beit_large_patch16_224" \
    dataset.transform_cfg.train.transform_mode="imagenet" \
    name=ttnet_sota_v1_beit_large_patch16_224 \
    logging.wandb.tags="[sota_v1,image_encoder]"

# swin_large_patch4_window7_224
python train.py experiment=sota_v1 \
    model.image_encoder="swin_large_patch4_window7_224" \
    dataset.transform_cfg.train.transform_mode="imagenet" \
    name=ttnet_sota_v1_swin_large_patch4_window7_224 \
    logging.wandb.tags="[sota_v1,image_encoder]"

# vit_large_patch16_224
python train.py experiment=sota_v1 \
    model.image_encoder="vit_large_patch16_224" \
    dataset.transform_cfg.train.transform_mode="imagenet" \
    name=ttnet_sota_v1_vit_large_patch16_224 \
    logging.wandb.tags="[sota_v1,image_encoder]"

# clip RN101
python train.py experiment=sota_v1 \
    model.image_encoder="RN101" \
    name=ttnet_sota_v1_clip_rn101 \
    logging.wandb.tags="[sota_v1,image_encoder]"

# clip RN50x4
python train.py experiment=sota_v1 \
    model.image_encoder="RN50x4" \
    name=ttnet_sota_v1_clip_rn50x4 \
    logging.wandb.tags="[sota_v1,image_encoder]"

# clip ViT-B/32
python train.py experiment=sota_v1 \
    model.image_encoder="ViT-B/32" \
    name=ttnet_sota_v1_clip_vit_b_32 \
    logging.wandb.tags="[sota_v1,image_encoder]"

# clip ViT-B/16
python train.py experiment=sota_v1 \
    model.image_encoder="ViT-B/16" \
    name=ttnet_sota_v1_clip_vit_b_16 \
    logging.wandb.tags="[sota_v1,image_encoder]"
