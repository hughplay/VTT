# ====================
# Base
# ====================

# python train.py experiment=sota_v5_base \
#     name=ttnet_sota_v5_base \
#     logging.wandb.tags="[sota_v5]"

# # ====================
# # Base +diff
# # ====================

# python train.py experiment=sota_v5_diff \
#     name=ttnet_sota_v5_diff_early \
#     model.diff_mode=early \
#     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_diff \
#     name=ttnet_sota_v5_diff_late \
#     model.diff_mode=late \
#     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_diff \
#     name=ttnet_sota_v5_diff_early_and_late \
#     model.diff_mode=early_and_late \
#     logging.wandb.tags="[sota_v5]"

# ====================
# Base +diff +mtm
# ====================

# # reuse sotv_v4
# # python train.py experiment=sota_v5_mtm \
# #     name=ttnet_sota_v5_mtm_0.15_all_zero \
# #     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_mtm \
#     name=ttnet_sota_v5_mtm_0.15_all_mask \
#     model.learned_mask=true \
#     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_mtm \
#     model.sample_mask_prob=0.5 \
#     name=ttnet_sota_v5_mtm_0.15_0.5_zero \
#     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_mtm \
#     model.sample_mask_prob=0.25 \
#     name=ttnet_sota_v5_mtm_0.15_0.25_zero \
#     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_mtm \
#     model.sample_mask_prob=0.75 \
#     name=ttnet_sota_v5_mtm_0.15_0.75_zero \
#     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_mtm \
#     name=ttnet_sota_v5_mtm_0.1_all_zero \
#     model.mask_ratio=0.1 \
#     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_mtm \
#     name=ttnet_sota_v5_mtm_0.05_all_zero \
#     model.mask_ratio=0.05 \
#     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_mtm \
#     name=ttnet_sota_v5_mtm_0.2_all_zero \
#     model.mask_ratio=0.2 \
#     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_mtm \
#     name=ttnet_sota_v5_mtm_0.25_all_zero \
#     model.mask_ratio=0.25 \
#     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_mtm \
#     name=ttnet_sota_v5_mtm_0.25_all_random \
#     model.mask_ratio=0.25 \
#     model.zero_prob=0 \
#     model.random_prob=1. \
#     logging.wandb.tags="[sota_v5]"

# ====================
# Base +diff +classify
# ====================

# w_topic is always 1., chage the w_classify and w_category here

# python train.py experiment=sota_v5_classify \
#     name=ttnet_sota_v5_wclass_0.25_wcat_0.1 \
#     criterion.loss.w_classify=0.25 \
#     criterion.loss.w_category=0.1 \
#     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_classify \
#     name=ttnet_sota_v5_wclass_0.25_wcat_0.25 \
#     criterion.loss.w_classify=0.25 \
#     criterion.loss.w_category=0.25 \
#     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_classify \
#     name=ttnet_sota_v5_wclass_0.25_wcat_0.5 \
#     criterion.loss.w_classify=0.25 \
#     criterion.loss.w_category=0.5 \
#     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_classify \
#     name=ttnet_sota_v5_wclass_0.25_wcat_0.75 \
#     criterion.loss.w_classify=0.25 \
#     criterion.loss.w_category=0.75 \
#     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_classify \
#     name=ttnet_sota_v5_wclass_0.125_wcat_0.1 \
#     criterion.loss.w_classify=0.125 \
#     criterion.loss.w_category=0.1 \
#     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_classify \
#     name=ttnet_sota_v5_wclass_0.125_wcat_0.25 \
#     criterion.loss.w_classify=0.125 \
#     criterion.loss.w_category=0.25 \
#     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_classify \
#     name=ttnet_sota_v5_wclass_0.125_wcat_0.5 \
#     criterion.loss.w_classify=0.125 \
#     criterion.loss.w_category=0.5 \
#     logging.wandb.tags="[sota_v5]"

# python train.py experiment=sota_v5_classify \
#     name=ttnet_sota_v5_wclass_0.125_wcat_0.75 \
#     criterion.loss.w_classify=0.125 \
#     criterion.loss.w_category=0.75 \
#     logging.wandb.tags="[sota_v5]"

# ==============================
# Base +diff +classify +mtm
# ==============================

# python train.py experiment=sota_v5_full \
#     name=ttnet_sota_v5_0.15_0.25_zero_wclass_0.125_wcat_0.1 \
#     model.sample_mask_prob=0.25 \
#     model.mask_ratio=0.15 \
#     criterion.loss.w_classify=0.125 \
#     criterion.loss.w_category=0.1 \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_full \
#     name=ttnet_sota_v5_0.15_0.25_zero_wclass_0.125_wcat_0.25 \
#     model.sample_mask_prob=0.25 \
#     model.mask_ratio=0.15 \
#     criterion.loss.w_classify=0.125 \
#     criterion.loss.w_category=0.25 \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_full \
#     name=ttnet_sota_v5_0.15_0.25_zero_wclass_0.25_wcat_0.75 \
#     model.sample_mask_prob=0.25 \
#     model.mask_ratio=0.15 \
#     criterion.loss.w_classify=0.25 \
#     criterion.loss.w_category=0.75 \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_full \
#     name=ttnet_sota_v5_0.15_0.25_zero_wclass_0.25_wcat_0.25 \
#     model.sample_mask_prob=0.25 \
#     model.mask_ratio=0.15 \
#     criterion.loss.w_classify=0.25 \
#     criterion.loss.w_category=0.25 \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_full \
#     name=ttnet_sota_v5_0.25_all_zero_wclass_0.125_wcat_0.1 \
#     model.mask_ratio=0.25 \
#     criterion.loss.w_classify=0.125 \
#     criterion.loss.w_category=0.1 \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_full \
#     name=ttnet_sota_v5_0.25_all_zero_wclass_0.125_wcat_0.25 \
#     model.mask_ratio=0.25 \
#     criterion.loss.w_classify=0.125 \
#     criterion.loss.w_category=0.25 \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_full \
#     name=ttnet_sota_v5_0.25_all_zero_wclass_0.25_wcat_0.75 \
#     model.mask_ratio=0.25 \
#     criterion.loss.w_classify=0.25 \
#     criterion.loss.w_category=0.75 \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_full \
#     name=ttnet_sota_v5_0.25_all_zero_wclass_0.25_wcat_0.25 \
#     model.mask_ratio=0.25 \
#     criterion.loss.w_classify=0.25 \
#     criterion.loss.w_category=0.25 \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_full \
#     name=ttnet_sota_v5_0.15_0.5_zero_wclass_0.125_wcat_0.1 \
#     model.sample_mask_prob=0.5 \
#     model.mask_ratio=0.15 \
#     criterion.loss.w_classify=0.125 \
#     criterion.loss.w_category=0.1 \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_full \
#     name=ttnet_sota_v5_0.15_0.5_zero_wclass_0.25_wcat_0.25 \
#     model.sample_mask_prob=0.5 \
#     model.mask_ratio=0.15 \
#     criterion.loss.w_classify=0.25 \
#     criterion.loss.w_category=0.25 \
#     logging.wandb.tags="[sota_v5,full]"

# *final full model
# python train.py experiment=sota_v5_full \
#     name=ttnet_sota_v5_0.15_0.5_zero_wclass_0.25_wcat_0.1 \
#     model.sample_mask_prob=0.5 \
#     model.mask_ratio=0.15 \
#     criterion.loss.w_classify=0.25 \
#     criterion.loss.w_category=0.1 \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_full \
#     name=ttnet_sota_v5_0.15_0.5_zero_wclass_0.125_wcat_0.5 \
#     model.sample_mask_prob=0.5 \
#     model.mask_ratio=0.15 \
#     criterion.loss.w_classify=0.125 \
#     criterion.loss.w_category=0.5 \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_full \
#     name=ttnet_sota_v5_0.15_0.5_zero_wclass_0.125_wcat_0.1 \
#     model.sample_mask_prob=0.5 \
#     model.mask_ratio=0.15 \
#     criterion.loss.w_classify=0.125 \
#     criterion.loss.w_category=0.1 \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_full \
#     name=ttnet_sota_v5_0.2_all_zero_wclass_0.125_wcat_0.1 \
#     model.mask_ratio=0.2 \
#     criterion.loss.w_classify=0.125 \
#     criterion.loss.w_category=0.1 \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_full \
#     name=ttnet_sota_v5_0.15_all_zero_wclass_0.125_wcat_0.1 \
#     model.mask_ratio=0.15 \
#     criterion.loss.w_classify=0.125 \
#     criterion.loss.w_category=0.1 \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_full \
#     name=ttnet_sota_v5_0.1_all_zero_wclass_0.125_wcat_0.1 \
#     model.mask_ratio=0.1 \
#     criterion.loss.w_classify=0.125 \
#     criterion.loss.w_category=0.1 \


# # ====================
# # Base +classify
# # ====================

# python train.py experiment=sota_v5_classify_only \
#     name=ttnet_sota_v5_wclass_0.25_wcat_0.1_no_diff \
#     criterion.loss.w_classify=0.25 \
#     criterion.loss.w_category=0.1 \
#     logging.wandb.tags="[sota_v5,key]"

# # ====================
# # Base +mtm
# # ====================

# python train.py experiment=sota_v5_mtm_only \
#     name=ttnet_sota_v5_0.15_0.5_zero_no_diff \
#     model.sample_mask_prob=0.5 \
#     model.mask_ratio=0.15 \
#     logging.wandb.tags="[sota_v5,key]"

# # ====================
# # Base +mtm +classify
# # ====================

# python train.py experiment=sota_v5_classify_only \
#     name=ttnet_sota_v5_0.15_0.5_zero_wclass_0.25_wcat_0.1_no_diff \
#     model.sample_mask_prob=0.5 \
#     model.mask_ratio=0.15 \
#     criterion.loss.w_classify=0.25 \
#     criterion.loss.w_category=0.1 \
#     logging.wandb.tags="[sota_v5,key]"

# # ====================
# # classify ablation
# # ====================

# python train.py experiment=sota_v5_final \
#     name=ttnet_sota_v5_final_wclass_1_wcat_0_w_topic_0.25 \
#     criterion.loss.w_classify=1. \
#     criterion.loss.w_category=0. \
#     criterion.loss.w_topic=0.25 \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_final \
#     name=ttnet_sota_v5_final_wclass_1_wcat_0.0625_w_topic_0 \
#     criterion.loss.w_classify=1. \
#     criterion.loss.w_category=0.0625 \
#     criterion.loss.w_topic=0. \
#     logging.wandb.tags="[sota_v5,full]"

# # ====================
# # difference ablation
# # ====================

# python train.py experiment=sota_v5_final \
#     name=ttnet_sota_v5_final_early \
#     model.diff_mode="early" \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_final \
#     name=ttnet_sota_v5_final_early_and_late \
#     model.diff_mode="early_and_late" \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_final \
#     name=ttnet_sota_v5_final_early_only \
#     model.diff_mode="early" \
#     model.diff_only=true \
#     logging.wandb.tags="[sota_v5,full]"

# python train.py experiment=sota_v5_final \
#     name=ttnet_sota_v5_final_late_only \
#     model.diff_mode="late" \
#     model.diff_only=true \
#     logging.wandb.tags="[sota_v5,full]"

# # ====================
# # mask ablation
# # ====================

# python train.py experiment=sota_v5_final \
#     name=ttnet_sota_v5_mask_0.05 \
#     model.mask_ratio=0.05 \
#     logging.wandb.tags="[sota_v5,mask]"

# python train.py experiment=sota_v5_final \
#     name=ttnet_sota_v5_mask_0.1 \
#     model.mask_ratio=0.1 \
#     logging.wandb.tags="[sota_v5,mask]"

# python train.py experiment=sota_v5_final \
#     name=ttnet_sota_v5_mask_0.2 \
#     model.mask_ratio=0.2 \
#     logging.wandb.tags="[sota_v5,mask]"

# python train.py experiment=sota_v5_final \
#     name=ttnet_sota_v5_mask_0.25 \
#     model.mask_ratio=0.25 \
#     logging.wandb.tags="[sota_v5,mask]"

# python train.py experiment=sota_v5_final \
#     name=ttnet_sota_v5_mask_0.3 \
#     model.mask_ratio=0.3 \
#     logging.wandb.tags="[sota_v5,mask]"

# ====================
# sample mask ablation
# ====================

python train.py experiment=sota_v5_final \
    name=ttnet_sota_v5_sample_mask_0.25 \
    model.sample_mask_prob=0.25 \
    logging.wandb.tags="[sota_v5,sample_mask]"

python train.py experiment=sota_v5_final \
    name=ttnet_sota_v5_sample_mask_0.75 \
    model.sample_mask_prob=0.75 \
    logging.wandb.tags="[sota_v5,sample_mask]"

python train.py experiment=sota_v5_final \
    name=ttnet_sota_v5_sample_mask_1 \
    model.sample_mask_prob=1. \
    logging.wandb.tags="[sota_v5,sample_mask]"
