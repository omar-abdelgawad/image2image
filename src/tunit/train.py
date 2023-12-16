# pylint: skip-file

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm import trange
from typing import Dict

from tunit.dataset import create_dataset
from tunit.utils import save_checkpoint, load_checkpoint, save_some_examples
from tunit import cfg
from tunit.generator import Generator
from tunit.discriminator import Discriminator
from tunit.guiding_network import GuidingNetwork
from tunit.utils import calc_contrastive_loss, calc_iic_loss

# _WRITER = SummaryWriter("runs/expirement_1")


def train_fn(
    disc: Discriminator,
    gen: Generator,
    gn: GuidingNetwork,
    loader: DataLoader,
):
    optimizer_gn = optim.Adam(
        params=gn.parameters(), lr=cfg.LEARNING_RATE, weight_decay=0.001
    )
    optimizer_gen = optim.RMSprop(
        params=gen.parameters(), lr=cfg.LEARNING_RATE, weight_decay=0.001
    )
    optimizer_disc = optim.RMSprop(
        params=disc.parameters(), lr=cfg.LEARNING_RATE, weight_decay=0.001
    )

    gen.train()
    disc.train()
    gn.train()

    train_it = iter(loader)
    t_train = trange(0, 1000, initial=0, total=1000)

    for i in t_train:
        try:
            imgs, _ = next(train_it)
        except:
            train_it = iter(loader)
            imgs, _ = next(train_it)

        x_org = imgs[0]
        x_tf = imgs[1]

        x_ref_idx = torch.randperm(x_org.size(0))

        x_org = x_org.cuda(0)
        x_tf = x_tf.cuda(0)
        x_ref_idx = x_ref_idx.cuda(0)

        x_ref = x_org.clone()
        x_ref = x_ref[x_ref_idx]

        q_cont = gn.moco(x_org)
        k_cont = k_cont.detach()

        q_disc = gn.iic(x_org)
        k_disc = gn.iic(x_tf)

        q_disc = torch.softmax(q_disc, 1)
        k_disc = torch.softmax(k_disc, 1)
