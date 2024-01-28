""" Trainer class"""
from pathlib import Path

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from img2img.cfg import DatasetType
from img2img.data import create_dataset
from img2img.models.pix2pix.generator import Generator
from img2img.models.pix2pix.discriminator import Discriminator
from img2img.models.pix2pix.utils import (
    save_checkpoint,
    load_checkpoint,
    save_some_examples,
)
from img2img.utils import prepare_sub_directories


class Pix2PixTrainer:
    def __init__(
        self,
        load_model: bool,
        learning_rate: float,
        train_batch_size: int,
        val_batch_size: int,
        device: str,
        path: str | Path,
        num_workers: int,
        train_dataset_path: str | Path,
        val_dataset_path: str | Path,
        chosen_dataset: DatasetType,
        l1_lambda: float,
    ) -> None:
        self.device = device
        self.l1_lambda = l1_lambda
        weights_path, self.eval_path = prepare_sub_directories(path)
        self.checkpoint_gen = weights_path / "gen.pth.tar"
        self.checkpoint_disc = weights_path / "disc.pth.tar"
        self._WRITER = SummaryWriter("runs/expirement_1")
        self.disc = Discriminator(in_channels=3).to(self.device)
        self.gen = Generator(in_channels=3).to(self.device)
        self.opt_disc = optim.Adam(
            self.disc.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.999),
        )
        self.opt_gen = optim.Adam(
            self.gen.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.999),
        )
        self.binary_cross_with_logits_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        if load_model:
            load_checkpoint(
                self.checkpoint_gen,
                self.gen,
                self.opt_gen,
                learning_rate,
            )
            load_checkpoint(
                self.checkpoint_disc,
                self.disc,
                self.opt_disc,
                learning_rate,
            )
        train_dataset = create_dataset(
            root_dir=train_dataset_path, dataset_type=chosen_dataset
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()

        val_dataset = create_dataset(
            root_dir=val_dataset_path, dataset_type=chosen_dataset
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=val_batch_size, shuffle=False
        )

    def train(self, num_epochs: int, save_model: bool, checkpoint_period: int) -> None:
        for epoch in range(num_epochs):
            print("Epoch:", epoch)
            self.train_fn(epoch)
            if save_model and epoch % checkpoint_period == 0:
                save_checkpoint(self.gen, self.opt_gen, filename=self.checkpoint_gen)
                save_checkpoint(self.disc, self.opt_disc, filename=self.checkpoint_disc)
            save_some_examples(
                self.gen,
                self.val_loader,
                epoch,
                folder=self.eval_path,
                writer=self._WRITER,
            )
        self._WRITER.close()

    def train_fn(
        self,
        epoch: int,
    ) -> None:
        """Process one epoch of training."""
        loop = tqdm(self.train_loader, leave=True)

        for idx, (x, y) in enumerate(loop):
            x, y = x.to(self.device), y.to(self.device)
            cur_stage = epoch * len(self.train_loader) + idx
            # train discriminator
            with torch.cuda.amp.autocast():
                y_fake = self.gen(x)
                disc_real_out = self.disc(x, y)
                disc_fake_out = self.disc(x, y_fake.detach())

                disc_real_loss = self.binary_cross_with_logits_loss(
                    disc_real_out, torch.ones_like(disc_real_out)
                )
                disc_fake_loss = self.binary_cross_with_logits_loss(
                    disc_fake_out, torch.zeros_like(disc_fake_out)
                )
                total_disc_loss = (disc_real_loss + disc_fake_loss) / 2
                self._WRITER.add_scalar("d_loss", total_disc_loss, cur_stage)

            self.disc.zero_grad(set_to_none=True)
            self.d_scaler.scale(total_disc_loss).backward()
            self.d_scaler.step(self.opt_disc)
            self.d_scaler.update()

            # train generator
            with torch.cuda.amp.autocast():
                disc_fake_out = self.disc(x, y_fake)
                gen_fake_loss = self.binary_cross_with_logits_loss(
                    disc_fake_out, torch.ones_like(disc_fake_out)
                )
                gen_l1_loss = self.l1_loss(y_fake, y) * self.l1_lambda
                total_gen_loss = gen_fake_loss + gen_l1_loss
                self._WRITER.add_scalar("g_loss", total_gen_loss, cur_stage)

            self.gen.zero_grad(set_to_none=True)
            self.g_scaler.scale(total_gen_loss).backward()
            self.g_scaler.step(self.opt_gen)
            self.g_scaler.update()
