# no mtm, topic, category
python train.py experiment=sota_v3 \
    model.mask_ratio=-1 \
    model.category_head=false \
    model.topic_head=false \
    criterion.loss.w_classify=null \
    criterion.loss.w_category=null \
    criterion.loss.w_topic=null \
    criterion.category=false \
    criterion.topic=false \
    name=ttnet_sota_v3_no_mtm_no_category_no_topic \
    logging.wandb.tags="[sota_v3,multitask]"

# w/ mtm
python train.py experiment=sota_v3 \
    model.mask_ratio=0.15 \
    model.category_head=false \
    model.topic_head=false \
    criterion.loss.w_classify=null \
    criterion.loss.w_category=null \
    criterion.loss.w_topic=null \
    criterion.category=false \
    criterion.topic=false \
    name=ttnet_sota_v3_w_mtm_no_category_no_topic \
    logging.wandb.tags="[sota_v3,multitask]"

# w/ category
python train.py experiment=sota_v3 \
    model.mask_ratio=-1 \
    model.category_head=true \
    model.topic_head=false \
    criterion.loss.w_classify=1. \
    criterion.loss.w_category=1. \
    criterion.loss.w_topic=null \
    criterion.category=true \
    criterion.topic=false \
    name=ttnet_sota_v3_no_mtm_w_category_no_topic \
    logging.wandb.tags="[sota_v3,multitask,mtm]"

# w/ topic
python train.py experiment=sota_v3 \
    model.mask_ratio=-1 \
    model.category_head=false \
    model.topic_head=true \
    criterion.loss.w_classify=1. \
    criterion.loss.w_category=null \
    criterion.loss.w_topic=1. \
    criterion.category=false \
    criterion.topic=true \
    name=ttnet_sota_v3_no_mtm_no_category_w_topic \
    logging.wandb.tags="[sota_v3,multitask]"

# w/ topic, category
python train.py experiment=sota_v3 \
    model.mask_ratio=-1 \
    model.category_head=true \
    model.topic_head=true \
    criterion.loss.w_classify=1. \
    criterion.loss.w_category=1. \
    criterion.loss.w_topic=1. \
    criterion.category=true \
    criterion.topic=true \
    name=ttnet_sota_v3_no_mtm_w_category_w_topic \
    logging.wandb.tags="[sota_v3,multitask]"

# w/ mtm, category
python train.py experiment=sota_v3 \
    model.mask_ratio=0.15 \
    model.category_head=true \
    model.topic_head=false \
    criterion.loss.w_classify=1. \
    criterion.loss.w_category=1. \
    criterion.loss.w_topic=null \
    criterion.category=true \
    criterion.topic=false \
    name=ttnet_sota_v3_w_mtm_w_category_no_topic \
    logging.wandb.tags="[sota_v3,multitask]"

# w/ mtm, topic
python train.py experiment=sota_v3 \
    model.mask_ratio=0.15 \
    model.category_head=false \
    model.topic_head=true \
    criterion.loss.w_classify=1. \
    criterion.loss.w_category=null \
    criterion.loss.w_topic=1. \
    criterion.category=false \
    criterion.topic=true \
    name=ttnet_sota_v3_w_mtm_no_category_w_topic \
    logging.wandb.tags="[sota_v3,multitask]"

# w/ mtm, topic, category
python train.py experiment=sota_v3 \
    model.mask_ratio=0.15 \
    model.category_head=true \
    model.topic_head=true \
    criterion.loss.w_classify=1. \
    criterion.loss.w_category=1. \
    criterion.loss.w_topic=1. \
    criterion.category=true \
    criterion.topic=true \
    name=ttnet_sota_v3_w_mtm_w_category_w_topic \
    logging.wandb.tags="[sota_v3,multitask]"

# no mtm, topic, category, ViT-B/32
python train.py experiment=sota_v3 \
    model.image_encoder="ViT-B/32" \
    model.mask_ratio=-1 \
    model.category_head=false \
    model.topic_head=false \
    criterion.loss.w_classify=null \
    criterion.loss.w_category=null \
    criterion.loss.w_topic=null \
    criterion.category=false \
    criterion.topic=false \
    name=ttnet_sota_v3_sm_no_mtm_no_category_no_topic \
    logging.wandb.tags="[sota_v3,multitask,sm]"

# w/ mtm, ViT-B/32
python train.py experiment=sota_v3 \
    model.image_encoder="ViT-B/32" \
    model.mask_ratio=0.15 \
    model.category_head=false \
    model.topic_head=false \
    criterion.loss.w_classify=null \
    criterion.loss.w_category=null \
    criterion.loss.w_topic=null \
    criterion.category=false \
    criterion.topic=false \
    name=ttnet_sota_v3_sm_w_mtm_no_category_no_topic \
    logging.wandb.tags="[sota_v3,multitask,sm]"

# w/ category, ViT-B/32
python train.py experiment=sota_v3 \
    model.image_encoder="ViT-B/32" \
    model.mask_ratio=-1 \
    model.category_head=true \
    model.topic_head=false \
    criterion.loss.w_classify=1. \
    criterion.loss.w_category=1. \
    criterion.loss.w_topic=null \
    criterion.category=true \
    criterion.topic=false \
    name=ttnet_sota_v3_sm_no_mtm_w_category_no_topic \
    logging.wandb.tags="[sota_v3,multitask,mtm,sm]"

# w/ topic, ViT-B/32
python train.py experiment=sota_v3 \
    model.image_encoder="ViT-B/32" \
    model.mask_ratio=-1 \
    model.category_head=false \
    model.topic_head=true \
    criterion.loss.w_classify=1. \
    criterion.loss.w_category=null \
    criterion.loss.w_topic=1. \
    criterion.category=false \
    criterion.topic=true \
    name=ttnet_sota_v3_sm_no_mtm_no_category_w_topic \
    logging.wandb.tags="[sota_v3,multitask,sm]"

# w/ topic, category, ViT-B/32
python train.py experiment=sota_v3 \
    model.image_encoder="ViT-B/32" \
    model.mask_ratio=-1 \
    model.category_head=true \
    model.topic_head=true \
    criterion.loss.w_classify=1. \
    criterion.loss.w_category=1. \
    criterion.loss.w_topic=1. \
    criterion.category=true \
    criterion.topic=true \
    name=ttnet_sota_v3_sm_no_mtm_w_category_w_topic \
    logging.wandb.tags="[sota_v3,multitask,sm]"

# w/ mtm, category, ViT-B/32
python train.py experiment=sota_v3 \
    model.image_encoder="ViT-B/32" \
    model.mask_ratio=0.15 \
    model.category_head=true \
    model.topic_head=false \
    criterion.loss.w_classify=1. \
    criterion.loss.w_category=1. \
    criterion.loss.w_topic=null \
    criterion.category=true \
    criterion.topic=false \
    name=ttnet_sota_v3_sm_w_mtm_w_category_no_topic \
    logging.wandb.tags="[sota_v3,multitask,sm]"

# w/ mtm, topic, ViT-B/32
python train.py experiment=sota_v3 \
    model.image_encoder="ViT-B/32" \
    model.mask_ratio=0.15 \
    model.category_head=false \
    model.topic_head=true \
    criterion.loss.w_classify=1. \
    criterion.loss.w_category=null \
    criterion.loss.w_topic=1. \
    criterion.category=false \
    criterion.topic=true \
    name=ttnet_sota_v3_sm_w_mtm_no_category_w_topic \
    logging.wandb.tags="[sota_v3,multitask,sm]"

# w/ mtm, topic, category, ViT-B/32
python train.py experiment=sota_v3 \
    model.image_encoder="ViT-B/32" \
    model.mask_ratio=0.15 \
    model.category_head=true \
    model.topic_head=true \
    criterion.loss.w_classify=1. \
    criterion.loss.w_category=1. \
    criterion.loss.w_topic=1. \
    criterion.category=true \
    criterion.topic=true \
    name=ttnet_sota_v3_sm_w_mtm_w_category_w_topic \
    logging.wandb.tags="[sota_v3,multitask,sm]"
