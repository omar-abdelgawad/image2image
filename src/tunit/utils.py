"""Utility functions for the model."""
from pathlib import Path

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torchvision.utils import make_grid

import cfg


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        self.a = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))
        self.b = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)


class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(FRN, self).__init__()
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(torch.mean(x**2, dim=[2, 3], keepdim=True) + self.eps)
        print(x.shape)
        print(self.gamma.shape)
        print(self.beta.shape)
        print(self.tau.shape)
        return torch.max(self.gamma * x + self.beta, self.tau)


# TODO: remove Magic numbers from this module
def save_some_examples(
    gen: nn.Module,
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    epoch: int,
    folder: Path,
    writer: SummaryWriter,
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
    gen.eval()
    with torch.inference_mode():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5
        x = x * 0.5 + 0.5
        x_concat = torch.cat([x, y_fake], dim=3)
        save_image(x_concat, folder / f"sample_{epoch}.png")
        img_grid = make_grid(x_concat)
        writer.add_image(f"test_image {epoch=}", img_grid)
        # save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        # save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 0:
            writer.add_graph(gen, x)
            save_image(y * 0.5 + 0.5, folder / f"label_{epoch}.png")
    gen.train()


@torch.inference_mode()
def evaluate_val_set(
    gen: nn.Module,
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    folder: Path,
) -> None:
    """Runs inference on all images in the val_loader and saves them in the folder.

    Args:
        gen (nn.Module): Generator model.
        val_loader (DataLoader): Dataloader for val set.
        folder (Path): Path for saving the images.
    """
    gen.eval()
    for idx, (x, y) in enumerate(val_loader):
        x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5
        x = x * 0.5 + 0.5
        y_concat = torch.cat([y, y_fake], dim=3)
        print(f"Saving {idx} image")
        save_image(y_concat, folder / f"val_{idx}.png")
    gen.train()


def save_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, filename: Path
) -> None:
    """Saves checkpoint for the model and optimizer in the folder filename.

    Args:
        model (nn.Module): torch Model.
        optimizer (optim.Optimizer): Optimizer.
        filename (Path): new File name/path.
    """
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(
    checkpoint_file: Path, model: nn.Module, optimizer: optim.Optimizer, lr: float
) -> None:
    """Loads checkpoint for the model and optimizer from the checkpoint_file.
    With the new learning rate.

    Args:
        checkpoint_file (Path): Saved model name/path.
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


def calc_iic_loss(x_out, x_tf_out, lamb=1.0, EPS=1e-10):
    # has had softmax applied
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert p_i_j.size() == (k, k)

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = -p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_j) - lamb * torch.log(p_i))

    loss = loss.sum()

    return loss


def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert x_tf_out.size(0) == bn and x_tf_out.size(1) == k

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.0  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def calc_contrastive_loss(query, key, queue, temp=0.07):
    N = query.shape[0]
    K = queue.shape[0]

    zeros = torch.zeros(N, dtype=torch.long).cuda(0)
    key = key.detach()
    logit_pos = torch.bmm(query.view(N, 1, -1), key.view(N, -1, 1))
    logit_neg = torch.mm(query.view(N, -1), queue.t().view(-1, K))

    logit = torch.cat([logit_pos.view(N, 1), logit_neg], dim=1)

    loss = torch.nn.CrossEntropyLoss(logit / temp, zeros)

    return loss


def calc_adv_loss(logit, mode):
    assert mode in ["d_real", "d_fake", "g"]
    if mode == "d_real":
        loss = F.relu(1.0 - logit).mean()
    elif mode == "d_fake":
        loss = F.relu(1.0 + logit).mean()
    else:
        loss = -logit.mean()

    return loss


def queue_data(data, k):
    return torch.cat([data, k], dim=0)


def dequeue_data(data, K=1024):
    if len(data) > K:
        return data[-K:]
    else:
        return data


def compute_grad_gp(d_out, x_in, is_patch=False):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum() if not is_patch else d_out.mean(),
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert grad_dout2.size() == x_in.size()
    reg = grad_dout2.sum() / batch_size
    return reg


def calc_recon_loss(predict, target):
    return torch.mean(torch.abs(predict - target))
