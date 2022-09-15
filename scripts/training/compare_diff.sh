# fuse diff, early
python train.py experiment=ttnet_diff \
    model.diff_mode=early \
    model.context_mode=fuse \
    name=ttnet_diff_early_fuse \
    logging.wandb.tags="[diff]"

# fuse diff, late
python train.py experiment=ttnet_diff \
    model.diff_mode=late \
    model.context_mode=fuse \
    name=ttnet_diff_late_fuse \
    logging.wandb.tags="[diff]"

# *attention diff, early, diff_first=false
python train.py experiment=ttnet_diff \
    model.diff_mode=early \
    model.context_mode=attention \
    model.diff_first=false \
    name=ttnet_diff_early_attention_last \
    logging.wandb.tags="[diff]"

# attention diff, early, diff_first=true
python train.py experiment=ttnet_diff \
    model.diff_mode=early \
    model.context_mode=attention \
    model.diff_first=true \
    name=ttnet_diff_early_attention_first \
    logging.wandb.tags="[diff]"

# *attention diff, late, diff_first=false
python train.py experiment=ttnet_diff \
    model.diff_mode=late \
    model.context_mode=attention \
    model.diff_first=false \
    name=ttnet_diff_late_attention_last \
    logging.wandb.tags="[diff]"

# *attention diff, late, diff_first=true
python train.py experiment=ttnet_diff \
    model.diff_mode=late \
    model.context_mode=attention \
    model.diff_first=true \
    name=ttnet_diff_late_attention_first \
    logging.wandb.tags="[diff]"

# attention diff, early_and_late, diff_first=false
python train.py experiment=ttnet_diff \
    model.diff_mode=early_and_late \
    model.context_mode=attention \
    name=ttnet_diff_both_attention \
    logging.wandb.tags="[diff]"

# attention cross, early, diff_first=true
python train.py experiment=ttnet_diff \
    model.diff_mode=early \
    model.context_mode=cross \
    model.diff_first=true \
    name=ttnet_diff_early_cross_first \
    logging.wandb.tags="[diff]"

# attention cross, late, diff_first=true
python train.py experiment=ttnet_diff \
    model.diff_mode=late \
    model.context_mode=cross \
    model.diff_first=true \
    name=ttnet_diff_late_cross_first \
    logging.wandb.tags="[diff]"

# attention cross, early, diff_first=false
python train.py experiment=ttnet_diff \
    model.diff_mode=early \
    model.context_mode=cross \
    model.diff_first=false \
    name=ttnet_diff_early_cross_last \
    logging.wandb.tags="[diff]"

# attention cross, late, diff_first=false
python train.py experiment=ttnet_diff \
    model.diff_mode=late \
    model.context_mode=cross \
    model.diff_first=false \
    name=ttnet_diff_late_cross_last \
    logging.wandb.tags="[diff]"
