# Base CST
python train.py experiment=baseline_cst

# Base GLACNet
python train.py experiment=baseline_glacnet

# Base TTNet (ViT-B/32)
# python train.py experiment=baseline_ttnet
python train.py experiment=baseline_ttnet_context \
    model.decoder_context_fusion=add

# Base TTNet (ViT-L/14)
python train.py experiment=baseline_ttnet_context \
    model.decoder_context_fusion=add \
    model.image_encoder="ViT-L/14" name="baseline_ttnet_ViT-L/14"
