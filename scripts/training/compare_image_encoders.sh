export CUDA_VISIBLE_DEVICES=6

# TTNet with ResNet152
python train.py experiment=baseline_ttnet_context \
    model.image_encoder="resnet152" name="baseline_ttnet_resnet152" \
    dataset.transform_cfg.train.transform_mode="imagenet" \
    logging.wandb.tags="[encoder]"

# TTNet with InceptionV3
python train.py experiment=baseline_ttnet_context \
    model.image_encoder="inception_v3" name="baseline_ttnet_inception_v3" \
    dataset.transform_cfg.train.n_px=299 \
    dataset.transform_cfg.train.transform_mode="imagenet" \
    logging.wandb.tags="[encoder]"

# TTNet with RN101
python train.py experiment=baseline_ttnet_context \
    model.image_encoder="RN101" name="baseline_ttnet_RN101" \
    logging.wandb.tags="[encoder]"

# TTNet with ViT-B/16
python train.py experiment=baseline_ttnet_context \
    model.image_encoder="ViT-B/16" name="baseline_ttnet_ViT-B/16" \
    logging.wandb.tags="[encoder]"

# TTNet with ViT-L/14
python train.py experiment=baseline_ttnet_context \
    model.image_encoder="ViT-L/14" name="baseline_ttnet_ViT-L/14" \
    logging.wandb.tags="[encoder]"

# TTNet with RN50x4
python train.py experiment=baseline_ttnet_context \
    model.image_encoder="RN50x4" name="baseline_ttnet_RN50x4" \
    dataset.transform_cfg.train.n_px=288 \
    logging.wandb.tags="[encoder]"

# TTNet with RN50x64
python train.py experiment=baseline_ttnet_context \
    model.image_encoder="RN50x64" name="baseline_ttnet_RN50x64" \
    dataset.transform_cfg.train.n_px=448 \
    logging.wandb.tags="[encoder]"

# TTNet with ViT-L/14@336px
python train.py experiment=baseline_ttnet_context \
    model.image_encoder="ViT-L/14@336px" name="baseline_ttnet_ViT-L/14@336px" \
    dataset.transform_cfg.train.n_px=336 \
    logging.wandb.tags="[encoder]"

# TTNet with beit_large_patch16_224
python train.py experiment=baseline_ttnet_context \
    model.image_encoder="beit_large_patch16_224" \
    name="baseline_ttnet_beit_large_patch16_224" \
    logging.wandb.tags="[encoder]"

# TTNet with swin_large_patch4_window7_224
python train.py experiment=baseline_ttnet_context \
    model.image_encoder="swin_large_patch4_window7_224" \
    name="baseline_ttnet_swin_large_patch4_window7_224" \
    logging.wandb.tags="[encoder]"

# TTNet with vit_large_patch16_224
python train.py experiment=baseline_ttnet_context \
    model.image_encoder="vit_large_patch16_224" \
    name="baseline_ttnet_vit_large_patch16_224" \
    logging.wandb.tags="[encoder]"
