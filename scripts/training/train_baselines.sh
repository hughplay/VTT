# Base CST
python train.py experiment=baseline_cst pl_trainer.devices='[0]'

# Base GLACNet
python train.py experiment=baseline_glacnet pl_trainer.devices='[1]'

# Base TTNet
python train.py experiment=baseline_ttnet pl_trainer.devices='[2]'
