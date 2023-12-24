import os
import math
from pathlib import Path
import yaml
import time

from torch.utils.data import DataLoader
from torch.nn import init
from torch.optim import lr_scheduler

from unit import cfg
from unit.data import create_dataset

# NOTE: The original implementation included vgg16 loss but we are not using it.


def get_data_loaders():
    """Returns the data loaders for training and validation."""
    train_dataset = create_dataset(cfg.TRAIN_DATASET_PATH, cfg.DatasetType.Edges2Shoes)
    val_dataset = create_dataset(cfg.VAL_DATASET_PATH, cfg.DatasetType.Edges2Shoes)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )
    return train_loader, val_loader


def prepare_sub_directories(path: str | Path) -> None:
    """Creates subdirectories for saving images and checkpoints."""
    path = Path(path)
    os.makedirs(path, exist_ok=True)
    os.makedirs(path / "evaluation", exist_ok=True)
    os.makedirs(path / "last_trained_weights", exist_ok=True)
    os.makedirs(path / "saved_models", exist_ok=True)


def weights_init(init_type: str = "gaussian"):
    """Initializes weights for the model."""

    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find("Conv") == 0 or classname.find("Linear") == 0) and hasattr(
            m, "weight"
        ):
            # print m.__class__.__name__
            if init_type == "gaussian":
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))  # type: ignore
            elif init_type == "default":
                pass
            else:
                assert 0, f"Unsupported initialization: {init_type}"
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    """Returns a learning rate scheduler."""
    if "lr_policy" not in hyperparameters or hyperparameters["lr_policy"] == "constant":
        scheduler = None  # constant scheduler
    elif hyperparameters["lr_policy"] == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=hyperparameters["step_size"],
            gamma=hyperparameters["gamma"],
            last_epoch=iterations,
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", hyperparameters["lr_policy"]
        )
    return scheduler


# Get model list for resume
def get_model_list(dirname, key):
    """Returns the latest model from a directory."""
    if os.path.exists(dirname) is False:
        return None
    gen_models = [
        os.path.join(dirname, f)
        for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f
    ]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


# def get_config(config: str | Path):
#     """Returns the configuration from yaml file."""
#     config = Path(config)
#     with config.open(encoding="utf-8") as stream:
#         return yaml.safe_load(stream)


class Timer:
    """A simple timer."""

    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.start_time is not None:
            print(self.msg % (time.perf_counter() - self.start_time))
