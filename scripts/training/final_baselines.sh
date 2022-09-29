# Base CST
python train.py experiment=baseline_cst \
    name="baseline_cst" \
    dataset.transform_cfg.train.transform_mode="imagenet" \
    logging.wandb.tags="[final_base]"

# CST, ViT-L/14
python train.py experiment=baseline_cst \
    name="baseline_cst_clip" \
    model.image_encoder="ViT-L/14" \
    dataset.transform_cfg.train.transform_mode="clip" \
    logging.wandb.tags="[final_base]"

# CST, ViT-L/14, shared text decoder
python train.py experiment=baseline_cst_shared \
    name="baseline_cst_shared_clip" \
    model.image_encoder="ViT-L/14" \
    dataset.transform_cfg.train.transform_mode="clip" \
    logging.wandb.tags="[final_base]"

# Base GLACNet
python train.py experiment=baseline_glacnet \
    name="baseline_glacnet" \
    dataset.transform_cfg.train.transform_mode="imagenet" \
    logging.wandb.tags="[final_base]"

# GLACNet, ViT-L/14
python train.py experiment=baseline_glacnet \
    name="baseline_glacnet_clip" \
    dataset.transform_cfg.train.transform_mode="clip" \
    model.image_encoder="ViT-L/14" \
    logging.wandb.tags="[final_base]"

# Base DenseCap
python train.py experiment=baseline_densecap \
    name="baseline_densecap" \
    logging.wandb.tags="[final_base]"

# densecap normalize weight
python train.py experiment=baseline_densecap \
    model.normalize_weight=true \
    name="baseline_densecap_norm" \
    logging.wandb.tags="[final_base]"
