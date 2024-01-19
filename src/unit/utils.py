import os
import math
from pathlib import Path

# import yaml
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision.utils import save_image
from torchvision.utils import make_grid

from unit import cfg
from unit.data import create_dataset

# NOTE: The original implementation included vgg16 loss but we are not using it.


# TODO: remove Magic numbers from this module
# TODO: unsupervised gen should cycle from x to y and vice versa
def save_some_examples(
    trainer: nn.Module,
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    epoch: int,
    folder: Path,
    # writer: SummaryWriter,
) -> None:
    """Saves a grid of generated images. Also saves ground truth if epoch is 0.

    Args:
        gen (nn.Module): Generator model.
        val_loader (DataLoader): Dataloader for train/val set.
        epoch (int): Current epoch.
        folder (Path): Folder to save the images in.
    """
    # TODO: refactor this function for single responsibility and improving readability
    x, y = next(iter(val_loader))
    x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
    trainer.eval()
    with torch.inference_mode():
        image_outputs = trainer.sample(x, y)
        image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]
        image_tensor = torch.cat(image_outputs, dim=0)
        image_grid = make_grid(image_tensor, normalize=True)
        save_image(image_grid, folder / f"sample_{epoch}.png")
        # writer.add_image(f"test_image {epoch=}", make_grid(x_concat))
        # if epoch == 0:
        #     writer.add_graph(trainer, x)
        #     save_image(y * 0.5 + 0.5, folder / f"label_{epoch}.png")
    trainer.train()


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
    """Creates subdirectories for saving images and checkpoints.
    Creates the directory given +
        evaluation subdirectory
        last_trained_weights subdirectory
        saved_models subdirectory
    """
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


def get_scheduler(optimizer, lr_policy, step_size=None, gamma=None, iterations=-1):
    """Returns a learning rate scheduler."""
    if lr_policy is None or lr_policy == "constant":
        scheduler = None
    elif lr_policy == "step":
        if step_size is None or gamma is None:
            raise ValueError
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            last_epoch=iterations,
        )
    else:
        raise NotImplementedError(
            f"learning rate policy [{lr_policy}] is not implemented"
        )
    return scheduler


# Get model list for resume
def get_model_list(dirname, key):
    """Returns the latest model from a directory."""
    if os.path.exists(dirname) is False:
        raise ValueError(f"{dirname} does not exist")
    gen_models = [
        os.path.join(dirname, f)
        for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f
    ]
    if not gen_models:
        raise ValueError(f"No correct model extension '.pt' found in {dirname}")
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


# def get_config(config: str | Path):
#     """Returns the configuration from yaml file."""
#     config = Path(config)
#     with config.open(encoding="utf-8") as stream:
#         return yaml.safe_load(stream)
