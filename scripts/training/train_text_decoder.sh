# baselines

# python train.py experiment=wikihow_text \
#     model.encoder_name="ViT-B/32" \
#     model.num_decoder_layers=2 \
#     logging.wandb.name="vit-b32-t2" \
#     logging.wandb.tags="[wikihow]"

# python train.py experiment=wikihow_text \
#     model.encoder_name="ViT-B/16" \
#     model.num_decoder_layers=2 \
#     logging.wandb.name="vit-b16-t2" \
#     logging.wandb.tags="[wikihow]"

# python train.py experiment=wikihow_text \
#     model.encoder_name="ViT-L/14" \
#     model.num_decoder_layers=2 \
#     logging.wandb.name="vit-l14-t2" \
#     logging.wandb.tags="[wikihow]"

# python train.py experiment=wikihow_text \
#     model.encoder_name="ViT-B/32" \
#     model.num_decoder_layers=4 \
#     logging.wandb.name="vit-b32-t4" \
#     logging.wandb.tags="[wikihow]"

# python train.py experiment=wikihow_text \
#     model.encoder_name="ViT-B/16" \
#     model.num_decoder_layers=4 \
#     logging.wandb.name="vit-b16-t4" \
#     logging.wandb.tags="[wikihow]"

# python train.py experiment=wikihow_text \
#     model.encoder_name="ViT-L/14" \
#     model.num_decoder_layers=4 \
#     logging.wandb.name="vit-l14-t4" \
#     logging.wandb.tags="[wikihow]"

# more training steps

python train.py experiment=wikihow_text \
    model.encoder_name="ViT-B/32" \
    model.num_decoder_layers=4 \
    logging.wandb.name="vit-b32-t4" \
    logging.wandb.tags="[wikihow]" \
    pl_trainer.max_epochs=300

python train.py experiment=wikihow_text \
    model.encoder_name="ViT-B/16" \
    model.num_decoder_layers=4 \
    logging.wandb.name="vit-b16-t4" \
    logging.wandb.tags="[wikihow]" \
    pl_trainer.max_epochs=300

python train.py experiment=wikihow_text \
    model.encoder_name="ViT-L/14" \
    model.num_decoder_layers=4 \
    logging.wandb.name="vit-l14-t4" \
    logging.wandb.tags="[wikihow]" \
    pl_trainer.max_epochs=300

# compare learning rate schedule

python train.py experiment=wikihow_text \
    model.encoder_name="ViT-B/32" \
    model.num_decoder_layers=4 \
    logging.wandb.name="vit-b32-t4" \
    logging.wandb.tags="[wikihow, lr_decay]" \
    scheduler="linear_warmup" \
    pl_trainer.max_epochs=300

python train.py experiment=wikihow_text \
    model.encoder_name="ViT-B/16" \
    model.num_decoder_layers=4 \
    logging.wandb.name="vit-b16-t4" \
    logging.wandb.tags="[wikihow, lr_decay]" \
    scheduler="linear_warmup" \
    pl_trainer.max_epochs=300

python train.py experiment=wikihow_text \
    model.encoder_name="ViT-L/14" \
    model.num_decoder_layers=4 \
    logging.wandb.name="vit-l14-t4" \
    logging.wandb.tags="[wikihow, lr_decay]" \
    scheduler="linear_warmup" \
    pl_trainer.max_epochs=300
