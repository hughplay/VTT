import dotenv
from hydra import compose, initialize

from src.utils.exptool import print_config, register_omegaconf_resolver

register_omegaconf_resolver()

dotenv.load_dotenv(override=True)


def test_conf():
    with initialize(config_path="../conf"):
        cfg = compose(config_name="train")

    print_config(cfg)

    assert cfg.dataset
    assert cfg.pipeline
    assert cfg.model
    assert cfg.criterion
    assert cfg.optim
    assert cfg.scheduler
    assert cfg.pl_trainer
    assert cfg.callbacks
    assert cfg.logging
