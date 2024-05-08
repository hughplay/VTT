# ====================
# Base +diff +classify
# ====================

# w_topic is always 1., chage the w_classify and w_category here

python train.py experiment=sota_v5_classify \
    name=ttnet_sm_sota_v5_wclass_0.25_wcat_1 \
    model.image_encoder="ViT-B/32" \
    criterion.loss.w_classify=0.25 \
    criterion.loss.w_category=1. \
    logging.wandb.tags="[sota_v5,sm]"

python train.py experiment=sota_v5_classify \
    name=ttnet_sm_sota_v5_wclass_0.5_wcat_1 \
    model.image_encoder="ViT-B/32" \
    criterion.loss.w_classify=0.5 \
    criterion.loss.w_category=1. \
    logging.wandb.tags="[sota_v5,sm]"

python train.py experiment=sota_v5_classify \
    name=ttnet_sm_sota_v5_wclass_0.75_wcat_1 \
    model.image_encoder="ViT-B/32" \
    criterion.loss.w_classify=0.75 \
    criterion.loss.w_category=1. \
    logging.wandb.tags="[sota_v5,sm]"

python train.py experiment=sota_v5_classify \
    name=ttnet_sm_sota_v5_wclass_1_wcat_1 \
    model.image_encoder="ViT-B/32" \
    criterion.loss.w_classify=1. \
    criterion.loss.w_category=1. \
    logging.wandb.tags="[sota_v5,sm]"

python train.py experiment=sota_v5_classify \
    name=ttnet_sm_sota_v5_wclass_0.25_wcat_0 \
    model.image_encoder="ViT-B/32" \
    criterion.loss.w_classify=0.25 \
    criterion.loss.w_category=0. \
    logging.wandb.tags="[sota_v5,sm]"

python train.py experiment=sota_v5_classify \
    name=ttnet_sm_sota_v5_wclass_0.5_wcat_0 \
    model.image_encoder="ViT-B/32" \
    criterion.loss.w_classify=0.5 \
    criterion.loss.w_category=0. \
    logging.wandb.tags="[sota_v5,sm]"

python train.py experiment=sota_v5_classify \
    name=ttnet_sm_sota_v5_wclass_0.75_wcat_0 \
    model.image_encoder="ViT-B/32" \
    criterion.loss.w_classify=0.75 \
    criterion.loss.w_category=0. \
    logging.wandb.tags="[sota_v5,sm]"

python train.py experiment=sota_v5_classify \
    name=ttnet_sm_sota_v5_wclass_1_wcat_0 \
    model.image_encoder="ViT-B/32" \
    criterion.loss.w_classify=1 \
    criterion.loss.w_category=0. \
    logging.wandb.tags="[sota_v5,sm]"

python train.py experiment=sota_v5_classify \
    name=ttnet_sm_sota_v5_wclass_0.25_wcat_0.1 \
    model.image_encoder="ViT-B/32" \
    criterion.loss.w_classify=0.25 \
    criterion.loss.w_category=0.1 \
    logging.wandb.tags="[sota_v5,sm]"

python train.py experiment=sota_v5_classify \
    name=ttnet_sm_sota_v5_wclass_0.25_wcat_0.25 \
    model.image_encoder="ViT-B/32" \
    criterion.loss.w_classify=0.25 \
    criterion.loss.w_category=0.25 \
    logging.wandb.tags="[sota_v5,sm]"

python train.py experiment=sota_v5_classify \
    name=ttnet_sm_sota_v5_wclass_0.25_wcat_0.5 \
    model.image_encoder="ViT-B/32" \
    criterion.loss.w_classify=0.25 \
    criterion.loss.w_category=0.5 \
    logging.wandb.tags="[sota_v5,sm]"

python train.py experiment=sota_v5_classify \
    name=ttnet_sm_sota_v5_wclass_0.25_wcat_0.75 \
    model.image_encoder="ViT-B/32" \
    criterion.loss.w_classify=0.25 \
    criterion.loss.w_category=0.75 \
    logging.wandb.tags="[sota_v5,sm]"

python train.py experiment=sota_v5_classify \
    name=ttnet_sm_sota_v5_wclass_0.125_wcat_0.1 \
    model.image_encoder="ViT-B/32" \
    criterion.loss.w_classify=0.125 \
    criterion.loss.w_category=0.1 \
    logging.wandb.tags="[sota_v5,sm]"

python train.py experiment=sota_v5_classify \
    name=ttnet_sm_sota_v5_wclass_0.125_wcat_0.25 \
    model.image_encoder="ViT-B/32" \
    criterion.loss.w_classify=0.125 \
    criterion.loss.w_category=0.25 \
    logging.wandb.tags="[sota_v5,sm]"

python train.py experiment=sota_v5_classify \
    name=ttnet_sm_sota_v5_wclass_0.125_wcat_0.5 \
    model.image_encoder="ViT-B/32" \
    criterion.loss.w_classify=0.125 \
    criterion.loss.w_category=0.5 \
    logging.wandb.tags="[sota_v5,sm]"

python train.py experiment=sota_v5_classify \
    name=ttnet_sm_sota_v5_wclass_0.125_wcat_0.75 \
    model.image_encoder="ViT-B/32" \
    criterion.loss.w_classify=0.125 \
    criterion.loss.w_category=0.75 \
    logging.wandb.tags="[sota_v5,sm]"
