# multi-task: +category
python train.py experiment=ttnet_multitask_classify \
    criterion.loss.w_category=1. criterion.loss.w_topic=0. \
    name=ttnet_multitask_category

# multi-task: +topic
python train.py experiment=ttnet_multitask_classify \
    criterion.loss.w_category=0. criterion.loss.w_topic=1. \
    name=ttnet_multitask_topic

# multi-task: +category +topic
python train.py experiment=ttnet_multitask_classify \
    criterion.loss.w_category=1. criterion.loss.w_topic=1.

# multi-task: +reconstruction
python train.py experiment=ttnet_multitask_reconstruction

# multi-task: +category +topic +reconstruction
python train.py experiment=ttnet_multitask

# dropout for category classification
python train.py experiment=ttnet_multitask_classify \
    criterion.loss.w_category=1. criterion.loss.w_topic=0. \
    model.reconstruction_head=false model.head_dropout=0.3 \
    name=ttnet_multitask_category_drop0.3 logging.wandb.tags="[multitask]"

# dropout for topic classification
python train.py experiment=ttnet_multitask_classify \
    criterion.loss.w_category=0. criterion.loss.w_topic=1. \
    model.reconstruction_head=false model.head_dropout=0.3 \
    name=ttnet_multitask_topic_drop0.3 logging.wandb.tags="[multitask]"

# dropout for category classification + topic classification
python train.py experiment=ttnet_multitask_classify \
    criterion.loss.w_category=1. criterion.loss.w_topic=1. \
    model.reconstruction_head=false model.head_dropout=0.3 \
    name=ttnet_multitask_drop0.3 logging.wandb.tags="[multitask]"
