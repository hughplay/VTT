python train.py experiment=sota_v4_diff \
    model.mask_ratio=-1 \
    model.topic_head=false \
    criterion.loss.w_classify=null \
    criterion.loss.w_topic=null \
    criterion.topic=false \
    name=ttnet_sota_v4_no_mtm_no_topic_w_diff \
    logging.wandb.tags="[sota_v4]"

python train.py experiment=sota_v4_diff \
    model.mask_ratio=0.15 \
    model.topic_head=false \
    criterion.loss.w_classify=null \
    criterion.loss.w_topic=null \
    criterion.topic=false \
    name=ttnet_sota_v4_w_mtm_no_topic_w_diff \
    logging.wandb.tags="[sota_v4]"

python train.py experiment=sota_v4_diff \
    model.mask_ratio=-1 \
    model.topic_head=true \
    criterion.loss.w_classify=1. \
    criterion.loss.w_topic=1. \
    criterion.topic=true \
    name=ttnet_sota_v4_no_mtm_w_topic_w_diff \
    logging.wandb.tags="[sota_v4]"

python train.py experiment=sota_v4_diff \
    model.mask_ratio=0.15 \
    model.topic_head=true \
    criterion.loss.w_classify=1. \
    criterion.loss.w_topic=1. \
    criterion.topic=true \
    name=ttnet_sota_v4_w_mtm_w_topic_w_diff \
    logging.wandb.tags="[sota_v4]"

python train.py experiment=sota_v4_diff \
    model.diff_mode=late \
    model.mask_ratio=-1 \
    model.topic_head=false \
    criterion.loss.w_classify=null \
    criterion.loss.w_topic=null \
    criterion.topic=false \
    name=ttnet_sota_v4_no_mtm_no_topic_w_diff_late \
    logging.wandb.tags="[sota_v4]"

python train.py experiment=sota_v4_diff \
    model.diff_mode=early \
    model.mask_ratio=-1 \
    model.topic_head=false \
    criterion.loss.w_classify=null \
    criterion.loss.w_topic=null \
    criterion.topic=false \
    name=ttnet_sota_v4_no_mtm_no_topic_w_diff_early \
    logging.wandb.tags="[sota_v4]"

python train.py experiment=sota_v4_diff \
    model.diff_mode=late \
    model.mask_ratio=0.15 \
    model.topic_head=false \
    criterion.loss.w_classify=null \
    criterion.loss.w_topic=null \
    criterion.topic=false \
    name=ttnet_sota_v4_w_mtm_no_topic_w_diff_late \
    logging.wandb.tags="[sota_v4]"

python train.py experiment=sota_v4_diff \
    model.diff_mode=late \
    model.mask_ratio=-1 \
    model.topic_head=true \
    criterion.loss.w_classify=1. \
    criterion.loss.w_topic=1. \
    criterion.topic=true \
    name=ttnet_sota_v4_no_mtm_w_topic_w_diff_late \
    logging.wandb.tags="[sota_v4]"

python train.py experiment=sota_v4_diff \
    model.diff_mode=late \
    model.mask_ratio=0.15 \
    model.topic_head=true \
    criterion.loss.w_classify=1. \
    criterion.loss.w_topic=1. \
    criterion.topic=true \
    name=ttnet_sota_v4_w_mtm_w_topic_w_diff_late \
    logging.wandb.tags="[sota_v4]"
