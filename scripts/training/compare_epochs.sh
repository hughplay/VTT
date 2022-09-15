export CUDA_VISIBLE_DEVICES=1

# epochs=10
python train.py experiment=baseline_ttnet_context \
    model.decoder_context_fusion=add pl_trainer.max_epochs=10 \
    name=baseline_ttnet_epoch10 logging.wandb.tags="[epochs]"

# epochs=25
python train.py experiment=baseline_ttnet_context \
    model.decoder_context_fusion=add pl_trainer.max_epochs=25 \
    name=baseline_ttnet_epoch25 logging.wandb.tags="[epochs]"

# epochs=100
python train.py experiment=baseline_ttnet_context \
    model.decoder_context_fusion=add pl_trainer.max_epochs=100 \
    name=baseline_ttnet_epoch100 logging.wandb.tags="[epochs]"
