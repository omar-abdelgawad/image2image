from pathlib import Path

# import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch.optim import lr_scheduler
from torchvision.utils import save_image
from torchvision.utils import make_grid

from img2img import cfg
from img2img.models.unit.trainer import UNIT_Trainer

# NOTE: The original implementation included vgg16 loss but we are not using it.


# TODO: remove Magic numbers from this module
# TODO: unsupervised gen should cycle from x to y and vice versa
def save_some_examples(
    trainer: UNIT_Trainer,
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    epoch: int,
    folder: Path,
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


# def get_config(config: str | Path):
#     """Returns the configuration from yaml file."""
#     config = Path(config)
#     with config.open(encoding="utf-8") as stream:
#         return yaml.safe_load(stream)
