import os
import torch
from torch import nn

from unit.utils import weights_init, get_model_list, get_scheduler

from unit import cfg
from unit.generator import VAEGen
from unit.discriminator import MsImageDis


class UNIT_Trainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gen_a = VAEGen(cfg.CHANNELS_IMG)  # fill rest of the arguments later gen
        self.gen_b = VAEGen(cfg.CHANNELS_IMG)  # fill rest of the arguments later gen
        self.dis_a = MsImageDis(cfg.CHANNELS_IMG)
        self.dis_b = MsImageDis(cfg.CHANNELS_IMG)
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # setup the optimizers
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam(
            [p for p in dis_params if p.requires_grad],
            lr=cfg.LEARNING_RATE,
            betas=cfg.BETA_OPTIM,
            weight_decay=cfg.WEIGHT_DECAY,
        )
        self.gen_opt = torch.optim.Adam(
            [p for p in gen_params if p.requires_grad],
            lr=cfg.LEARNING_RATE,
            betas=cfg.BETA_OPTIM,
            weight_decay=cfg.WEIGHT_DECAY,
        )
        self.dis_scheduler = get_scheduler(
            self.dis_opt, cfg.LR_POLICY, cfg.STEP_SIZE, cfg.GAMMA
        )
        self.gen_scheduler = get_scheduler(
            self.dis_opt, cfg.LR_POLICY, cfg.STEP_SIZE, cfg.GAMMA
        )

        # Network initialization
        self.apply(weights_init(cfg.INIT))
        self.dis_a.apply(weights_init("gaussian"))
        self.dis_b.apply(weights_init("gaussian"))

    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    # TODO: replace this reconstruction loss with L1Loss
    def recon_criterion(self, input_tens, target_tens):
        return torch.mean(torch.abs(input_tens - target_tens))

    # TODO: what did he do here? why is he summing the returning the average of mu squared?
    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, x_a, x_b):
        self.gen_opt.zero_grad()

        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = (
            self.gen_a.decode(h_a_recon + n_a_recon)
            if cfg.RECONSTRUCTION_X_CYC_WEIGHT > 0
            else None
        )
        x_bab = (
            self.gen_b.decode(h_b_recon + n_b_recon)
            if cfg.RECONSTRUCTION_X_CYC_WEIGHT > 0
            else None
        )
        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # # domain-invariant perceptual loss
        # self.loss_gen_vgg_a = (
        #     self.compute_vgg_loss(self.vgg, x_ba, x_b)
        #     if hyperparameters["vgg_w"] > 0
        #     else 0
        # )
        # self.loss_gen_vgg_b = (
        #     self.compute_vgg_loss(self.vgg, x_ab, x_a)
        #     if hyperparameters["vgg_w"] > 0
        #     else 0
        # )
        # total loss
        self.loss_gen_total = (
            cfg.GAN_WEIGHT * self.loss_gen_adv_a
            + cfg.GAN_WEIGHT * self.loss_gen_adv_b
            + cfg.RECONSTRUCTION_X_WEIGHT * self.loss_gen_recon_x_a
            + cfg.RECONSTRUCTION_KL_WEIGHT * self.loss_gen_recon_kl_a
            + cfg.RECONSTRUCTION_X_WEIGHT * self.loss_gen_recon_x_b
            + cfg.RECONSTRUCTION_KL_WEIGHT * self.loss_gen_recon_kl_b
            + cfg.RECONSTRUCTION_X_CYC_WEIGHT * self.loss_gen_cyc_x_a
            + cfg.RECONSTRUCTION_KL_CYC_WEIGHT * self.loss_gen_recon_kl_cyc_aba
            + cfg.RECONSTRUCTION_X_CYC_WEIGHT * self.loss_gen_cyc_x_b
            + cfg.RECONSTRUCTION_KL_CYC_WEIGHT * self.loss_gen_recon_kl_cyc_bab
            # + hyperparameters["vgg_w"] * self.loss_gen_vgg_a
            # + hyperparameters["vgg_w"] * self.loss_gen_vgg_b
        )
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    # TODO: remove the extra calculation of x_ba and x_ab if they are going to get detached anyway.
    def dis_update(self, x_a, x_b):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = (
            cfg.GAN_WEIGHT * self.loss_dis_a + cfg.GAN_WEIGHT * self.loss_dis_b
        )
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict["a"])
        self.gen_b.load_state_dict(state_dict["b"])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict["a"])
        self.dis_b.load_state_dict(state_dict["b"])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
        self.dis_opt.load_state_dict(state_dict["dis"])
        self.gen_opt.load_state_dict(state_dict["gen"])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(
            self.dis_opt,
            cfg.LR_POLICY,
            cfg.STEP_SIZE,
            cfg.GAMMA,
            iterations,
        )
        self.gen_scheduler = get_scheduler(
            self.gen_opt, cfg.LR_POLICY, cfg.STEP_SIZE, cfg.GAMMA, iterations
        )
        print(f"Resume from iteration {iterations}")
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, "gen_%08d.pt" % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, "dis_%08d.pt" % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, "optimizer.pt")
        torch.save(
            {"a": self.gen_a.state_dict(), "b": self.gen_b.state_dict()}, gen_name
        )
        torch.save(
            {"a": self.dis_a.state_dict(), "b": self.dis_b.state_dict()}, dis_name
        )
        torch.save(
            {"gen": self.gen_opt.state_dict(), "dis": self.dis_opt.state_dict()},
            opt_name,
        )
