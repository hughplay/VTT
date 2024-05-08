# TTNet with ResNet152
python train.py experiment=sota_v5_final \
    model.image_encoder="resnet152" name="ttnet_sota_v5_resnet152" \
    dataset.transform_cfg.train.transform_mode="imagenet" \
    logging.wandb.tags="[sota_v5,full,encoder]"

# TTNet with InceptionV3
python train.py experiment=sota_v5_final \
    model.image_encoder="inception_v3" name="ttnet_sota_v5_inception_v3" \
    dataset.transform_cfg.train.transform_mode="imagenet" \
    logging.wandb.tags="[sota_v5,full,encoder]"

# TTNet with beit_large_patch16_224
python train.py experiment=sota_v5_final \
    model.image_encoder="beit_large_patch16_224" \
    name="ttnet_sota_v5_beit_large_patch16_224" \
    dataset.transform_cfg.train.transform_mode="imagenet" \
    logging.wandb.tags="[sota_v5,full,encoder]"

# TTNet with swin_large_patch4_window7_224
python train.py experiment=sota_v5_final \
    model.image_encoder="swin_large_patch4_window7_224" \
    name="ttnet_sota_v5_swin_large_patch4_window7_224" \
    dataset.transform_cfg.train.transform_mode="imagenet" \
    logging.wandb.tags="[sota_v5,full,encoder]"

# TTNet with vit_large_patch16_224
python train.py experiment=sota_v5_final \
    model.image_encoder="vit_large_patch16_224" \
    name="ttnet_sota_v5_vit_large_patch16_224" \
    dataset.transform_cfg.train.transform_mode="imagenet" \
    logging.wandb.tags="[sota_v5,full,encoder]"

# TTNet with RN50
python train.py experiment=sota_v5_final \
    model.image_encoder="RN50" name="ttnet_sota_v5_RN50" \
    logging.wandb.tags="[sota_v5,full,encoder]"

# TTNet with RN101
python train.py experiment=sota_v5_final \
    model.image_encoder="RN101" name="ttnet_sota_v5_RN101" \
    logging.wandb.tags="[sota_v5,full,encoder]"

# TTNet with ViT-B/32
python train.py experiment=sota_v5_final \
    model.image_encoder="ViT-B/32" name="ttnet_sota_v5_ViT-B-32" \
    logging.wandb.tags="[sota_v5,full,encoder]"

# TTNet with ViT-B/16
python train.py experiment=sota_v5_final \
    model.image_encoder="ViT-B/16" name="ttnet_sota_v5_ViT-B-16" \
    logging.wandb.tags="[sota_v5,full,encoder]"
