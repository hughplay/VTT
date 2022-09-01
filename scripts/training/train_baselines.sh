# Base CST
python train.py experiment=baseline_cst

# Base GLACNet
python train.py experiment=baseline_glacnet

# Base TTNet
# python train.py experiment=baseline_ttnet
python train.py experiment=baseline_ttnet_context \
    model.decoder_context_fusion=add
