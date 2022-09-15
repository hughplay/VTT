# sota v2
python train.py experiment=sota_v2 \
    name=ttnet_sota_v2 \
    logging.wandb.tags="[sota_v2,multitask,baseline,mtm]"

# no mtm
python train.py experiment=sota_v2 \
    model.mask_ratio=-1 \
    name=ttnet_sota_v2_no_mtm \
    logging.wandb.tags="[sota_v2,multitask,baseline,mtm]"

# multitask ablation (default: w/ topic) for the large model (ViT-L/14)
# w/o topic, category
python train.py experiment=sota_v2 \
    model.topic_head=false \
    criterion.loss.w_classify=null \
    criterion.topic=false \
    name=ttnet_sota_v2_no_category_no_topic \
    logging.wandb.tags="[sota_v2,multitask]"

# w/ topic, category
python train.py experiment=sota_v2 \
    model.category_head=true \
    criterion.loss.w_category=1. \
    criterion.category=true \
    name=ttnet_sota_v2_topic_category \
    logging.wandb.tags="[sota_v2,multitask]"

# w/ category
python train.py experiment=sota_v2 \
    model.category_head=true \
    model.topic_head=false \
    criterion.loss.w_category=1. \
    criterion.loss.w_topic=null \
    criterion.category=true \
    criterion.topic=false \
    name=ttnet_sota_v2_category \
    logging.wandb.tags="[sota_v2,multitask]"

# no mtm, topic, category
python train.py experiment=sota_v2 \
    model.mask_ratio=-1 \
    model.topic_head=false \
    criterion.loss.w_classify=null \
    criterion.topic=false \
    name=ttnet_sota_v2_no_mtm_no_category_no_topic \
    logging.wandb.tags="[sota_v2,multitask]"

# multitask ablation (default: w/ topic) for the small model (ViT-B/32)
# w/ topic
python train.py experiment=sota_v2 \
    model.image_encoder="ViT-B/32" \
    name=ttnet_sota_v2_sm_topic \
    logging.wandb.tags="[sota_v2,multitask,image_encoder]"

# w/o topic, category
python train.py experiment=sota_v2 \
    model.image_encoder="ViT-B/32" \
    model.topic_head=false \
    criterion.loss.w_classify=null \
    criterion.topic=false \
    name=ttnet_sota_v2_sm_no_category_no_topic \
    logging.wandb.tags="[sota_v2,multitask]"

# w/ topic, category
python train.py experiment=sota_v2 \
    model.image_encoder="ViT-B/32" \
    model.category_head=true \
    criterion.loss.w_category=1. \
    criterion.category=true \
    name=ttnet_sota_v2_sm_topic_category \
    logging.wandb.tags="[sota_v2,multitask]"

# w/ category
python train.py experiment=sota_v2 \
    model.image_encoder="ViT-B/32" \
    model.category_head=true \
    model.topic_head=false \
    criterion.loss.w_category=1. \
    criterion.loss.w_topic=null \
    criterion.category=true \
    criterion.topic=false \
    name=ttnet_sota_v2_sm_category \
    logging.wandb.tags="[sota_v2,multitask]"

# image encoder ablation
# resnet152
python train.py experiment=sota_v2 \
    model.image_encoder="resnet152" \
    name=ttnet_sota_v2_resnet152 \
    dataset.transform_cfg.train.transform_mode="imagenet" \
    logging.wandb.tags="[sota_v2,image_encoder]"

# beit_large_patch16_224
python train.py experiment=sota_v2 \
    model.image_encoder="beit_large_patch16_224" \
    dataset.transform_cfg.train.transform_mode="imagenet" \
    name=ttnet_sota_v2_beit_large_patch16_224 \
    logging.wandb.tags="[sota_v2,image_encoder]"

# swin_large_patch4_window7_224
python train.py experiment=sota_v2 \
    model.image_encoder="swin_large_patch4_window7_224" \
    dataset.transform_cfg.train.transform_mode="imagenet" \
    name=ttnet_sota_v2_swin_large_patch4_window7_224 \
    logging.wandb.tags="[sota_v2,image_encoder]"

# vit_large_patch16_224
python train.py experiment=sota_v2 \
    model.image_encoder="vit_large_patch16_224" \
    dataset.transform_cfg.train.transform_mode="imagenet" \
    name=ttnet_sota_v2_vit_large_patch16_224 \
    logging.wandb.tags="[sota_v2,image_encoder]"

# clip RN50
python train.py experiment=sota_v2 \
    model.image_encoder="RN50" \
    name=ttnet_sota_v2_clip_rn50 \
    logging.wandb.tags="[sota_v2,image_encoder]"

# clip RN101
python train.py experiment=sota_v2 \
    model.image_encoder="RN101" \
    name=ttnet_sota_v2_clip_rn101 \
    logging.wandb.tags="[sota_v2,image_encoder]"

# clip RN50x4
python train.py experiment=sota_v2 \
    model.image_encoder="RN50x4" \
    name=ttnet_sota_v2_clip_rn50x4 \
    logging.wandb.tags="[sota_v2,image_encoder]"

# clip ViT-B/32 (multitask ablation)

# clip ViT-B/16
python train.py experiment=sota_v2 \
    model.image_encoder="ViT-B/16" \
    name=ttnet_sota_v2_clip_vit_b_16 \
    logging.wandb.tags="[sota_v2,image_encoder]"
