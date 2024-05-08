# random one
python test.py /log/exp/vtt/VTTDataModule.TTNetDiff.TellingLossV1.2022-09-19_21-44-42 --update_func test_miss_random_one --prefix miss_one --update_wandb

python test.py /log/exp/vtt/VTTDataModule.TTNetMTM.GenerationLoss.2022-09-16_10-59-03 --update_func test_miss_random_one --prefix miss_one --update_wandb

python test.py /log/exp/vtt/VTTDataModule.DenseCap.GenerationLoss.2022-09-25_15-15-34 --update_func test_miss_random_one --prefix miss_one --update_wandb

python test.py /log/exp/vtt/VTTDataModule.GLACNet.GenerationLoss.2022-09-18_17-56-36 --update_func test_miss_random_one --prefix miss_one --update_wandb

python test.py /log/exp/vtt/VTTDataModule.CST.GenerationLoss.2022-09-17_00-06-25 --update_func test_miss_random_one --prefix miss_one --update_wandb

python test.py /log/exp/vtt/VTTDataModule.TTNetDiff.TellingLossV1.2022-09-19_08-53-36 --update_func test_miss_random_one --prefix miss_one --update_wandb

# init fin
python test.py /log/exp/vtt/VTTDataModule.TTNetDiff.TellingLossV1.2022-09-19_21-44-42 --update_func test_miss_init_fin --prefix init_fin_only --update_wandb

python test.py /log/exp/vtt/VTTDataModule.TTNetMTM.GenerationLoss.2022-09-16_10-59-03 --update_func test_miss_init_fin --prefix init_fin_only --update_wandb

python test.py /log/exp/vtt/VTTDataModule.DenseCap.GenerationLoss.2022-09-25_15-15-34 --update_func test_miss_init_fin --prefix init_fin_only --update_wandb

python test.py /log/exp/vtt/VTTDataModule.GLACNet.GenerationLoss.2022-09-18_17-56-36 --update_func test_miss_init_fin --prefix init_fin_only --update_wandb

python test.py /log/exp/vtt/VTTDataModule.CST.GenerationLoss.2022-09-17_00-06-25 --update_func test_miss_init_fin --prefix init_fin_only --update_wandb

python test.py /log/exp/vtt/VTTDataModule.TTNetDiff.TellingLossV1.2022-09-19_08-53-36 --update_func test_miss_init_fin --prefix init_fin_only --update_wandb
