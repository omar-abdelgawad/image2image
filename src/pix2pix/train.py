"""Main script for training the model. Can train from scratch or resume from a checkpoint."""
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pix2pix.dataset import create_dataset
from pix2pix.utils import save_checkpoint, load_checkpoint, save_some_examples
from pix2pix import cfg
from pix2pix.generator import Generator
from pix2pix.discriminator import Discriminator

_WRITER = SummaryWriter("runs/expirement_1")


def train_fn(
    disc: Discriminator,
    gen: Generator,
    loader: DataLoader,
    opt_disc: optim.Optimizer,
    opt_gen: optim.Optimizer,
    l1: nn.L1Loss,
    bce: nn.BCEWithLogitsLoss,
    g_scaler,
    d_scaler,
    epoch: int,
) -> None:
    """Process one epoch of training.

    Args:
        disc (Discriminator): Pix2Pix Discriminator.
        gen (Generator): Pix2Pix Generator.
        loader (DataLoader): Train Data loader.
        opt_disc (optim.Optimizer): Discriminator optimizer.
        opt_gen (optim.Optimizer): Generator optimizer.
        l1 (nn.L1Loss): L1 loss for the generator.
        bce (nn.BCEWithLogitsLoss): Binary cross entropy loss for the discriminator.
        g_scaler (torch.cuda.amp.GradScaler): Gradient scaler for the generator.
        d_scaler (torch.cuda.amp.GradScaler): Gradient scaler for the discriminator.
        epoch (int): Epoch number.
    """
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
        cur_stage = epoch * len(loader) + idx
        # train discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            disc_real_out = disc(x, y)
            disc_fake_out = disc(x, y_fake.detach())

            disc_real_loss = bce(disc_real_out, torch.ones_like(disc_real_out))
            disc_fake_loss = bce(disc_fake_out, torch.zeros_like(disc_fake_out))
            total_disc_loss = (disc_real_loss + disc_fake_loss) / 2
            _WRITER.add_scalar("d_loss", total_disc_loss, cur_stage)

        disc.zero_grad()
        d_scaler.scale(total_disc_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # train generator
        with torch.cuda.amp.autocast():
            disc_fake_out = disc(x, y_fake)
            gen_fake_loss = bce(disc_fake_out, torch.ones_like(disc_fake_out))
            gen_l1_loss = l1(y_fake, y) * cfg.L_1_LAMBDA
            total_gen_loss = gen_fake_loss + gen_l1_loss
            _WRITER.add_scalar("g_loss", total_gen_loss, cur_stage)

        opt_gen.zero_grad()
        g_scaler.scale(total_gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()


def main() -> int:
    """Entry point for training loop."""
    disc = Discriminator(in_channels=3).to(cfg.DEVICE)
    gen = Generator(in_channels=3).to(cfg.DEVICE)
    opt_disc = optim.Adam(
        disc.parameters(),
        lr=cfg.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        gen.parameters(),
        lr=cfg.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    binary_cross_with_logits_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    if cfg.LOAD_MODEL:
        load_checkpoint(
            cfg.CHECKPOINT_GEN,
            gen,
            opt_gen,
            cfg.LEARNING_RATE,
        )
        load_checkpoint(
            cfg.CHECKPOINT_DISC,
            disc,
            opt_disc,
            cfg.LEARNING_RATE,
        )
    train_dataset = create_dataset(
        root_dir=cfg.TRAIN_DATASET_PATH, dataset_type=cfg.CHOSEN_DATASET
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    val_dataset = create_dataset(
        root_dir=cfg.VAL_DATASET_PATH, dataset_type=cfg.CHOSEN_DATASET
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.VAL_BATCH_SIZE, shuffle=False)

    for epoch in range(cfg.NUM_EPOCHS):
        print("Epoch:", epoch)
        train_fn(
            disc,
            gen,
            train_loader,
            opt_disc,
            opt_gen,
            l1_loss,
            binary_cross_with_logits_loss,
            g_scaler,
            d_scaler,
            epoch,
        )
        if cfg.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=cfg.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=cfg.CHECKPOINT_DISC)
        save_some_examples(
            gen, val_loader, epoch, folder=cfg.EVALUATION_PATH, writer=_WRITER
        )
    _WRITER.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
