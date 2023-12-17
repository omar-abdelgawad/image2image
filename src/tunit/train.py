# pylint: skip-file

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
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
from tunit.utils import (
    calc_contrastive_loss,
    calc_iic_loss,
    calc_adv_loss,
    compute_grad_gp,
    calc_recon_loss,
)

# _WRITER = SummaryWriter("runs/expirement_1")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def get_loader(args, dataset):
    train_dataset = dataset['train']
    val_dataset = dataset['val']
    if 'afhq' in args.dataset:
        val_dataset = dataset['val']['VAL']

    print(len(val_dataset))

    # GAN_IIC_SEMI
    if 0.0 < args.p_semi < 1.0:
        assert 'SEMI' in args.train_mode
        train_sup_dataset = train_dataset['SUP']
        train_unsup_dataset = train_dataset['UNSUP']

        if args.distributed:
            train_sup_sampler = torch.utils.data.distributed.DistributedSampler(train_sup_dataset)
            train_unsup_sampler = torch.utils.data.distributed.DistributedSampler(train_unsup_dataset)
        else:
            train_sup_sampler = None
            train_unsup_sampler = None

        # If there are not cpus enough, set workers to 0
        train_sup_loader = torch.utils.data.DataLoader(train_sup_dataset, batch_size=args.batch_size,
                                                      shuffle=(train_sup_sampler is None), num_workers=0,
                                                      pin_memory=True, sampler=train_sup_sampler, drop_last=False)
        train_unsup_loader = torch.utils.data.DataLoader(train_unsup_dataset, batch_size=args.batch_size,
                                                      shuffle=(train_unsup_sampler is None), num_workers=0,
                                                      pin_memory=True, sampler=train_unsup_sampler, drop_last=False)

        train_loader = {'SUP': train_sup_loader, 'UNSUP': train_unsup_loader}
        train_sampler = {'SUP': train_sup_sampler, 'UNSUP': train_unsup_sampler}

    # GAN_SUP / GAN_IIC_UN
    else:
        train_dataset_ = train_dataset['TRAIN']
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_)
        else:
            train_sampler = None
        train_loader = torch.utils.data.DataLoader(train_dataset_, batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None), num_workers=args.workers,
                                                   pin_memory=True, sampler=train_sampler, drop_last=False)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch, shuffle=True,
                                             num_workers=0, pin_memory=True, drop_last=False)

    val_loader = {'VAL': val_loader, 'VALSET': val_dataset if not args.dataset in ['afhq_cat', 'afhq_dog', 'afhq_wild'] else dataset['val']['FULL'], 'TRAINSET': train_dataset['FULL']}
    if 'afhq' in args.dataset:
        val_loader['IDX'] = train_dataset['IDX']

    return train_loader, val_loader, train_sampler

def trainGAN_UNSUP(
    disc: Discriminator,
    gen: Generator,
    gn: GuidingNetwork,
    loader: DataLoader,
    l1: nn.L1Loss,
    bce: nn.BCEWithLogitsLoss,
    g_scaler,
    d_scaler,
    epoch: int,
):
    iters = 1000
    den = iters // 1000
    unsup_start = 0
    separated = 65
    ema_start = 66
    fid_start = 66

    unsup_start = 0 // den
    separated = 65 // den
    ema_start = 66 // den
    fid_start = 66 // den

    # avg meter
    d_losses = AverageMeter()
    d_advs = AverageMeter()
    d_gps = AverageMeter()

    g_losses = AverageMeter()
    g_advs = AverageMeter()
    g_imgrecs = AverageMeter()
    g_styconts = AverageMeter()

    c_losses = AverageMeter()
    moco_losses = AverageMeter()
    iic_losses = AverageMeter()

    # set nets
    D = disc(
        in_channels=cfg.CHANNELS_IMG,
        channels_multiplier=cfg.CHANNELS_MULTIPLIER,
        out_channels=cfg.K,
    )
    G = gen(
        in_channels=cfg.CHANNELS_IMG,
        channels_multiplier=cfg.CHANNELS_MULTIPLIER,
        out_channels=cfg.CHANNELS_IMG,
    )
    G_EMA = gen(
        in_channels=cfg.CHANNELS_IMG,
        channels_multiplier=cfg.CHANNELS_MULTIPLIER,
        out_channels=cfg.CHANNELS_IMG,
    )
    C = gn(
        in_channels=cfg.CHANNELS_IMG,
        channels_multiplier=cfg.CHANNELS_MULTIPLIER,
        out_channels=cfg.K,
    )
    C_EMA = gn(
        in_channels=cfg.CHANNELS_IMG,
        channels_multiplier=cfg.CHANNELS_MULTIPLIER,
        out_channels=cfg.K,
    )
    # set opts
    d_opt = torch.optim.RMSprop(D.parameters(), 1e-4, weight_decay=0.0001)
    g_opt = torch.optim.RMSprop(G.parameters(), 1e-4, weight_decay=0.0001)
    c_opt = torch.optim.Adam(C.parameters(), 1e-4, weight_decay=0.001)
    # switch to train mode
    D.train()
    G.train()
    C.train()
    C_EMA().train()
    G_EMA().train()

    train_loader, val_loader, train_sampler = get_loader(args, {'train': train_dataset, 'val': val_dataset})
    queue_loader = train_loader['UNSUP'] if 0.0 < args.p_semi < 1.0 else train_loader
    queue = initialize_queue(C_EMA, 0, queue_loader, feat_size=args.sty_dim)

    # summary writer
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

        #################
        # BEGIN Train C #
        #################

        q_cont = C.moco(x_org)
        k_cont = k_cont.detach()

        q_disc = C.iic(x_org)
        k_disc = C.iic(x_tf)

        q_disc = F.softmax(q_disc, 1)
        k_disc = F.softmax(k_disc, 1)

        iic_loss = calc_iic_loss(q_disc, k_disc)
        moco_loss = calc_contrastive_loss(args, q_cont, k_cont, queue)
        c_loss = moco_loss + 5.0 * iic_loss

        if epoch >= 65:
            c_loss = 0.1 * c_loss

        c_opt.zero_grad()
        c_loss.backward()

        c_opt.step()
        ###############
        # END Train C #
        ###############

        ####################
        # BEGIN Train GANs #
        ####################
        if epoch >= 65:
            training_mode = "C2GANs"
            with torch.no_grad():
                q_disc = C.iic(x_org)
                y_org = torch.argmax(q_disc, 1)
                y_ref = y_org.clone()
                y_ref = y_ref[x_ref_idx]
                s_ref = C.moco(x_ref)
                c_src = G.enc(x_org)
                x_fake = G.dec(c_src, s_ref)

            x_ref.requires_grad_()

            d_real_logit, _ = D(x_ref, y_ref)
            d_fake_logit, _ = D(x_fake.detach(), y_ref)

            d_adv_real = calc_adv_loss(d_real_logit, "d_real")
            d_adv_fake = calc_adv_loss(d_fake_logit, "d_fake")

            d_adv = d_adv_real + d_adv_fake

            d_gp = 10.00 * compute_grad_gp(d_real_logit, x_ref, is_patch=False)

            d_loss = d_adv + d_gp

            d_opt.zero_grad()
            d_adv_real.backward(retain_graph=True)
            d_gp.backward()
            d_adv_fake.backward()

            d_opt.step()

            # Train G
            s_src = C.moco(x_org)
            s_ref = C.moco(x_ref)

            c_src = G.enc(x_org)
            x_fake = G.dec(c_src, s_ref)

            x_rec = G.dec(c_src, s_src)

            g_fake_logit, _ = D(x_fake, y_ref)
            g_rec_logit, _ = D(x_rec, y_org)

            g_adv_fake = calc_adv_loss(g_fake_logit, "g")
            g_adv_rec = calc_adv_loss(g_rec_logit, "g")

            g_adv = g_adv_fake + g_adv_rec

            g_imgrec = calc_recon_loss(x_rec, x_org)

            s_fake = C.moco(x_fake)

            g_sty_contrastive = calc_contrastive_loss(args, s_fake, s_ref_ema, queue)

            g_loss = 1.0 * g_adv + 0.1 * g_imgrec + 0.01 * g_sty_contrastive

            g_opt.zero_grad()
            c_opt.zero_grad()
            g_loss.backward()

            c_opt.step()
            g_opt.step()
        ##################
        # END Train GANs #
        ##################
