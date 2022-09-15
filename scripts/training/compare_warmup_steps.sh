export CUDA_VISIBLE_DEVICES=3

# warmup=100
python train.py experiment=baseline_ttnet_context \
    model.decoder_context_fusion=add scheduler.num_warmup_steps=200 \
    name=baseline_ttnet_warmup200 logging.wandb.tags="[warmup]"

# warmup=250
python train.py experiment=baseline_ttnet_context \
    model.decoder_context_fusion=add scheduler.num_warmup_steps=250 \
    name=baseline_ttnet_warmup250 logging.wandb.tags="[warmup]"

# *warmup=500
python train.py experiment=baseline_ttnet_context \
    model.decoder_context_fusion=add scheduler.num_warmup_steps=500 \
    name=baseline_ttnet_warmup500 logging.wandb.tags="[warmup]"

# warmup=1000
python train.py experiment=baseline_ttnet_context \
    model.decoder_context_fusion=add scheduler.num_warmup_steps=1000 \
    name=baseline_ttnet_warmup1000 logging.wandb.tags="[warmup]"

# warmup=4000
python train.py experiment=baseline_ttnet_context \
    model.decoder_context_fusion=add scheduler.num_warmup_steps=4000 \
    name=baseline_ttnet_warmup4000 logging.wandb.tags="[warmup]"
