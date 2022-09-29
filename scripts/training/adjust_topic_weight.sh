python train.py experiment=sota_v4_diff \
    model.diff_mode=late \
    model.mask_ratio=0.15 \
    model.topic_head=true \
    criterion.loss.w_classify=2. \
    criterion.loss.w_topic=1. \
    criterion.topic=true \
    name=ttnet_sota_v4_w_mtm_w_topic_w_diff_late_w2 \
    logging.wandb.tags="[sota_v4]"

python train.py experiment=sota_v4_diff \
    model.diff_mode=late \
    model.mask_ratio=0.15 \
    model.topic_head=true \
    criterion.loss.w_classify=4. \
    criterion.loss.w_topic=1. \
    criterion.topic=true \
    name=ttnet_sota_v4_w_mtm_w_topic_w_diff_late_w4 \
    logging.wandb.tags="[sota_v4]"

python train.py experiment=sota_v4_diff \
    model.diff_mode=late \
    model.mask_ratio=0.15 \
    model.topic_head=true \
    criterion.loss.w_classify=10. \
    criterion.loss.w_topic=1. \
    criterion.topic=true \
    name=ttnet_sota_v4_w_mtm_w_topic_w_diff_late_w10 \
    logging.wandb.tags="[sota_v4]"


python train.py experiment=sota_v4_diff \
    model.diff_mode=late \
    model.mask_ratio=0.15 \
    model.topic_head=true \
    criterion.loss.w_classify=100. \
    criterion.loss.w_topic=1. \
    criterion.topic=true \
    name=ttnet_sota_v4_w_mtm_w_topic_w_diff_late_w100 \
    logging.wandb.tags="[sota_v4]"

python train.py experiment=sota_v4_diff \
    model.diff_mode=late \
    model.mask_ratio=0.15 \
    model.topic_head=true \
    criterion.loss.w_classify=0.5 \
    criterion.loss.w_topic=1. \
    criterion.topic=true \
    name=ttnet_sota_v4_w_mtm_w_topic_w_diff_late_w0.5 \
    logging.wandb.tags="[sota_v4]"

python train.py experiment=sota_v4_diff \
    model.diff_mode=late \
    model.mask_ratio=0.15 \
    model.topic_head=true \
    criterion.loss.w_classify=0.25 \
    criterion.loss.w_topic=1. \
    criterion.topic=true \
    name=ttnet_sota_v4_w_mtm_w_topic_w_diff_late_w0.25 \
    logging.wandb.tags="[sota_v4]"

python train.py experiment=sota_v4_diff \
    model.diff_mode=late \
    model.mask_ratio=0.15 \
    model.topic_head=true \
    criterion.loss.w_classify=0.1 \
    criterion.loss.w_topic=1. \
    criterion.topic=true \
    name=ttnet_sota_v4_w_mtm_w_topic_w_diff_late_w0.1 \
    logging.wandb.tags="[sota_v4]"

python train.py experiment=sota_v4_diff \
    model.diff_mode=late \
    model.mask_ratio=0.15 \
    model.topic_head=true \
    criterion.loss.w_classify=0.01 \
    criterion.loss.w_topic=1. \
    criterion.topic=true \
    name=ttnet_sota_v4_w_mtm_w_topic_w_diff_late_w0.01 \
    logging.wandb.tags="[sota_v4]"

python train.py experiment=sota_v4_diff \
    model.diff_mode=late \
    model.mask_ratio=0.15 \
    model.topic_head=true \
    criterion.loss.w_classify=0.75 \
    criterion.loss.w_topic=1. \
    criterion.topic=true \
    name=ttnet_sota_v4_w_mtm_w_topic_w_diff_late_w0.75 \
    logging.wandb.tags="[sota_v4]"

python train.py experiment=sota_v4_diff \
    model.diff_mode=late \
    model.mask_ratio=0.15 \
    model.topic_head=true \
    criterion.loss.w_classify=0.125 \
    criterion.loss.w_topic=1. \
    criterion.topic=true \
    name=ttnet_sota_v4_w_mtm_w_topic_w_diff_late_w0.125 \
    logging.wandb.tags="[sota_v4]"
