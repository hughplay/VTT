export CUDA_VISIBLE_DEVICES=6

# TTNet with ResNet152
python train.py experiment=baseline_ttnet \
    model.image_encoder="resnet152" name="baseline_ttnet_resnet152" \
    pl_trainer.precision=32

# TTNet with InceptionV3
python train.py experiment=baseline_ttnet \
    model.image_encoder="inception_v3" name="baseline_ttnet_inception_v3" \
    dataset.transform_cfg.train.n_px=299 pl_trainer.precision=32

# TTNet with RN101
python train.py experiment=baseline_ttnet \
    model.image_encoder="RN101" name="baseline_ttnet_RN101"

# TTNet with ViT-B/16
python train.py experiment=baseline_ttnet \
    model.image_encoder="ViT-B/16" name="baseline_ttnet_ViT-B/16"

export CUDA_VISIBLE_DEVICES=7

# TTNet with ViT-L/14
python train.py experiment=baseline_ttnet \
    model.image_encoder="ViT-L/14" name="baseline_ttnet_ViT-L/14"

# TTNet with RN50x4
python train.py experiment=baseline_ttnet \
    model.image_encoder="RN50x4" name="baseline_ttnet_RN50x4" \
    dataset.transform_cfg.train.n_px=288

# TTNet with RN50x64
python train.py experiment=baseline_ttnet \
    model.image_encoder="RN50x64" name="baseline_ttnet_RN50x64" \
    dataset.transform_cfg.train.n_px=448

# TTNet with ViT-L/14@336px
python train.py experiment=baseline_ttnet \
    model.image_encoder="ViT-L/14@336px" name="baseline_ttnet_ViT-L/14@336px" \
    dataset.transform_cfg.train.n_px=336
