# 2 gpus
python train.py experiment=baseline_ttnet_context \
    model.image_encoder="ViT-L/14" name="baseline_ttnet_ViT-L/14_g2" \
    pl_trainer.devices="[1,2]" scheduler.num_warmup_steps=1000

# 4 gpus
python train.py experiment=baseline_ttnet_context \
    model.image_encoder="ViT-L/14" name="baseline_ttnet_ViT-L/14_g4" \
    pl_trainer.devices="[4,5,6,7]" scheduler.num_warmup_steps=500
