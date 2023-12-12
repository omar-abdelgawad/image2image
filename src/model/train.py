"""Main script for training the model. Can train from scratch or resume from a checkpoint."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.dataset import create_dataset
from model.utils import save_checkpoint, load_checkpoint, save_some_examples
from model import cfg
from model.generator import Generator
from model.discriminator import Discriminator

writer = SummaryWriter("runs/expirement_1")


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler, epoch: int
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
        cur_stage = epoch * len(loader) + idx
        # train discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())

            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
            writer.add_scalar("d_loss", D_loss, cur_stage)

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1(y_fake, y) * cfg.L_1_LAMBDA
            G_loss = G_fake_loss + L1
            writer.add_scalar("g_loss", G_loss, cur_stage)

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()


# TODO: Create a script for evaluation.
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
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

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
            L1_LOSS,
            BCE,
            g_scaler,
            d_scaler,
            epoch,
        )
        if cfg.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=cfg.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=cfg.CHECKPOINT_DISC)
        save_some_examples(
            gen, val_loader, epoch, folder=cfg.EVALUATION_PATH, writer=writer
        )
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
