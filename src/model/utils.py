"""Utility functions for the model."""
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model import cfg


def save_some_examples(
    gen: nn.Module, val_loader: DataLoader, epoch: int, folder: str
) -> None:
    """Saves a grid of generated images. Also saves ground truth if epoch is 0.

    Args:
        gen (nn.Module): Generator model.
        val_loader (DataLoader): Dataloader for train/val set.
        epoch (int): Current epoch.
        folder (str): Folder to save the images in.
    """
    x, y = next(iter(val_loader))
    x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5
        x = x * 0.5 + 0.5
        x_concat = torch.cat([x, y_fake], dim=3)
        save_image(x_concat, folder + f"/sample_{epoch}.png")
        # save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        # save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 0:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, filename: str
) -> None:
    """Saves checkpoint for the model and optimizer in the folder filename.

    Args:
        model (nn.Module): torch Model.
        optimizer (optim.Optimizer): Optimizer.
        filename (str): new File name/path.
    """
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(
    checkpoint_file: str, model: nn.Module, optimizer: optim.Optimizer, lr: float
) -> None:
    """Loads checkpoint for the model and optimizer from the checkpoint_file. With the new learning rate.

    Args:
        checkpoint_file (str): Saved model name/path.
        model (nn.Module): Model object to restore its state.
        optimizer (optim.Optimizer): Optimizer object to restore its state.
        lr (float): Learning rate.
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # if we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
