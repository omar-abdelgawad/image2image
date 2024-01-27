"""Generator model for T-UNIT."""
import torch
from torch import nn

# from torch.nn import init

# import math

from img2img.models.tunit.blocks import GenConvBlock
from img2img.models.tunit.blocks import GenResBlock

# from tunit.blocks import LinearBlock


class Generator(nn.Module):
    """Generator Class for T-UNIT model.

    Args:
        in_channels (int, optional): _description_. Defaults to 3.
        channels_multiplier: (int, optional): _description_. Defaults to 64.
        out_channels (int, optional): _description_. Defaults to 3.
        use_sn (bool, optional): _description_. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels_multiplier: int = 64,
        out_channels: int = 3,
        use_sn: bool = False,
    ) -> None:
        super().__init__()

        # self.nf_mlp = 256

        # self.decoder_norm = "adain"

        # self.adaptive_param_getter = get_num_adain_params
        # self.adaptive_param_assign = assign_adain_params

        self.enc = Encoder(
            in_channels=in_channels,
            channels_multiplier=channels_multiplier,
            out_channels=channels_multiplier * 8,
        )
        self.dec = Decoder(
            in_channels=channels_multiplier * 8,
            channels_multiplier=channels_multiplier,
            out_channels=out_channels,
            use_sn=use_sn,
        )

        # self.mlp = MLP(
        #     channels_multiplier,
        #     self.adaptive_param_getter(self.dec),
        #     self.nf_mlp,
        #     3,
        #     "none",
        #     "relu",
        # )

        # self.apply(weights_init("kaiming"))

    def forward(self, x_src: torch.Tensor, s_ref: torch.Tensor) -> torch.Tensor:
        """Forward pass for T-UNIT's generator. Downsamples the image then
        upsamples using convtranspose2d.

        Args:
            x_src (torch.Tensor): Batched input Image(s) tensor.
            s_ref (torch.Tensor): Batched style reference tensor.
        Returns:
            torch.Tensor: Batched output Image(s) tensor
        """
        return self.decode(self.enc(x_src), s_ref)

    # def decode(self, cnt, sty):
    #     adapt_params = self.mlp(sty)
    #     self.adaptive_param_assign(adapt_params, self.dec)
    #     out = self.dec(cnt)
    #     return out

    def _initialize_weights(self, mode="fan_in"):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()


class Encoder(nn.Module):
    """Encoder for T-UNIT's generator.

    Args:
        in_channels (int, optional): _description_. Defaults to 3.
        channels_multiplier: (int, optional): _description_. Defaults to 64.
        out_channels (int, optional): _description_. Defaults to 512.
        use_sn (bool, optional): _description_. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels_multiplier: int = 64,
        out_channels: int = 512,
        use_sn: bool = False,
    ) -> None:
        super().__init__()

        self.layer_1 = nn.Sequential(
            GenConvBlock(
                in_channels=in_channels,
                out_channels=channels_multiplier,
                kernel_size=7,
                stride=1,
                padding=3,
                padd_type="reflect",
                ins=True,
                use_sn=use_sn,
            ),
            nn.ReLU(inplace=True),
            GenConvBlock(
                in_channels=channels_multiplier,
                out_channels=channels_multiplier * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                padd_type="reflect",
                ins=True,
                use_sn=use_sn,
            ),
            nn.ReLU(inplace=True),
            GenConvBlock(
                in_channels=channels_multiplier * 2,
                out_channels=channels_multiplier * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                padd_type="reflect",
                ins=True,
                use_sn=use_sn,
            ),
            nn.ReLU(inplace=True),
            GenConvBlock(
                in_channels=channels_multiplier * 4,
                out_channels=channels_multiplier * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                padd_type="reflect",
                ins=True,
                use_sn=use_sn,
            ),
            nn.ReLU(inplace=True),
        )

        self.layer_2 = GenResBlock(
            in_channels=channels_multiplier * 8,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            ins=True,
            use_sn=use_sn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the encoder.

        Args:
            x (torch.Tensor): Batched input Image(s) tensor.
        Returns:
            torch.Tensor: Batched output Image(s) tensor
        """
        return self.layer_2(self.layer_1(x))


class Decoder(nn.Module):
    """Decoder for T-UNIT's generator.

    Args:
        in_channels (int, optional): _description_. Defaults to 512.
        channels_multiplier: (int, optional): _description_. Defaults to 64.
        out_channels (int, optional): _description_. Defaults to 3.
        use_sn (bool, optional): _description_. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int = 512,
        channels_multiplier: int = 64,
        out_channels: int = 3,
        use_sn: bool = False,
    ) -> None:
        super().__init__()

        self.padd = nn.ReflectionPad2d(padding=3)

        self.layer_1 = nn.Sequential(
            GenResBlock(
                in_channels=in_channels,
                out_channels=channels_multiplier * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                ins=False,
                use_sn=use_sn,
            ),
            GenResBlock(
                in_channels=in_channels,
                out_channels=channels_multiplier * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                ins=False,
                use_sn=use_sn,
            ),
        )

        self.layer_2 = nn.Sequential(
            GenConvBlock(
                in_channels=channels_multiplier * 8,
                out_channels=channels_multiplier * 4,
                kernel_size=5,
                stride=1,
                padding=2,
                ins=False,
                up=True,
                use_sn=use_sn,
            ),
            nn.ReLU(inplace=True),
            GenConvBlock(
                in_channels=channels_multiplier * 4,
                out_channels=channels_multiplier * 2,
                kernel_size=5,
                stride=1,
                padding=2,
                ins=False,
                up=True,
                use_sn=use_sn,
            ),
            nn.ReLU(inplace=True),
            GenConvBlock(
                in_channels=channels_multiplier * 2,
                out_channels=channels_multiplier,
                kernel_size=5,
                stride=1,
                padding=2,
                ins=False,
                up=True,
                use_sn=use_sn,
            ),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(
                in_channels=channels_multiplier,
                out_channels=out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
        )

        if use_sn:
            self.classifier = nn.utils.spectral_norm(self.classifier)

        self.classifier.add_module("tanh", nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the decoder.

        Args:
            x (torch.Tensor): Batched input Image(s) tensor.
        Returns:
            torch.Tensor: Batched output Image(s) tensor
        """
        return self.classifier(self.padd(self.layer_2(self.layer_1(x))))


# IMPLEMENTATION OF MLP FOR ADIN LOSS (PAPER'S APPROACH)
# class MLP(nn.Module):
#     def __init__(self, nf_in, nf_out, nf_mlp, num_blocks, norm, act, use_sn=False):
#         super(MLP, self).__init__()
#         self.model = nn.ModuleList()
#         nf = nf_mlp
#         self.model.append(LinearBlock(nf_in, nf, norm=norm, act=act, use_sn=use_sn))
#         for _ in range(num_blocks - 2):
#             self.model.append(LinearBlock(nf, nf, norm=norm, act=act, use_sn=use_sn))
#         self.model.append(
#             LinearBlock(nf, nf_out, norm="none", act="none", use_sn=use_sn)
#         )
#         self.model = nn.Sequential(*self.model)

#     def forward(self, x):
#         return self.model(x.view(x.size(0), -1))


# def weights_init(init_type="gaussian"):
#     def init_fun(m):
#         classname = m.__class__.__name__
#         if (classname.find("Conv") == 0 or classname.find("Linear") == 0) and hasattr(
#             m, "weight"
#         ):
#             if init_type == "gaussian":
#                 init.normal_(m.weight.data, 0.0, 0.02)
#             elif init_type == "xavier":
#                 init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
#             elif init_type == "kaiming":
#                 init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
#             elif init_type == "orthogonal":
#                 init.orthogonal_(m.weight.data, gain=math.sqrt(2))
#             elif init_type == "default":
#                 pass
#             else:
#                 assert 0, "Unsupported initialization: {}".format(init_type)
#             if hasattr(m, "bias") and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)

#     return init_fun


# def assign_adain_params(adain_params, model):
#     for m in model.modules():
#         if m.__class__.__name__ == "AdaIN2d":
#             mean = adain_params[:, : m.num_features]
#             std = adain_params[:, m.num_features : 2 * m.num_features]
#             m.bias = mean.contiguous().view(-1)
#             m.weight = std.contiguous().view(-1)
#             if adain_params.size(1) > 2 * m.num_features:
#                 adain_params = adain_params[:, 2 * m.num_features :]


# def get_num_adain_params(model):
#     num_adain_params = 0
#     for m in model.modules():
#         if m.__class__.__name__ == "AdaIN2d":
#             num_adain_params += 2 * m.num_features
#     return num_adain_params


if __name__ == "__main__":
    G = Generator(in_channels=3, channels_multiplier=64, out_channels=3)
    x_in = torch.randn(size=(32, 3, 128, 128))
    print("encoder", G.enc(x_in).shape)
    print("decoder", G.dec(G.enc(x_in)).shape)
