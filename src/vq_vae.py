from typing import List, Optional, Union

import torch
from diffusers.models.autoencoders.autoencoder_dc import (
    Decoder as DCDecoder,
)
from diffusers.models.autoencoders.autoencoder_dc import (
    Encoder as DCEncoder,
)
from diffusers.models.autoencoders.vae import (
    Decoder,
    Encoder,
)
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler


import torch.nn as nn
from einops import pack, rearrange, unpack
from torch import Tensor, int32
from torch.nn import Module

from typing import Tuple

import torch.nn.functional as F
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import (
    SpatialNorm,
)
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
)
from diffusers.models.normalization import RMSNorm, get_normalization
from diffusers.models.unets.unet_1d_blocks import (
    ResConvBlock,
    SelfAttention1d,
    Upsample1d,
    get_down_block,
)
from diffusers.utils import is_torch_version


class UpBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]

        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def forward(self, hidden_states, temb=None):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        hidden_states = self.up(hidden_states)
        return hidden_states


class UNetMidBlock1D(nn.Module):
    def __init__(
        self, mid_channels: int, in_channels: int, out_channels: Optional[int] = None
    ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        # there is always at least one resnet
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        for attn, resnet in zip(self.attentions, self.resnets):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        return hidden_states


class Encoder1D(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock1D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        double_z=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = torch.nn.Conv1d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock1D(
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv1d(
            block_out_channels[-1], conv_out_channels, 3, padding=1
        )

        self.gradient_checkpointing = False

    def forward(self, x):
        sample = x
        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            if is_torch_version(">=", "1.11.0"):
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )

                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, use_reentrant=False
                )
            else:
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample
                    )
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample
                )

        else:
            # down
            for down_block in self.down_blocks:
                sample = down_block(sample)[0]

            # middle
            sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class Decoder1D(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        norm_type="group",  # group, spatial
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv1d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock1D(
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, _ in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            up_block = UpBlock1D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
            )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv1d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, z, latent_embeds=None):
        sample = z
        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
                # sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds
                )
                # sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block), sample, latent_embeds
                    )
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            # sample = sample.to(upscale_dtype)
            # up
            for up_block in self.up_blocks:
                sample = up_block(sample, latent_embeds)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


def pixel_unshuffle(x, r):
    # Compute new shape
    N, C, Hr = x.shape
    H = Hr // r
    # Reshape and permute
    x = x.view(N, C, H, r)  # (N, C, H, r)
    x = x.permute(0, 1, 3, 2)  # (N, C, r, H)
    x = x.reshape(N, C * r, H)  # (N, C*r, H)
    return x


def pixel_shuffle(x, r):
    *batch_dims, cr, h = x.shape
    c = cr // r
    x = x.reshape(*batch_dims, c, r, h)  # (*, C, r, H)
    x = x.permute(
        *range(len(batch_dims)),
        0 + len(batch_dims),
        2 + len(batch_dims),
        1 + len(batch_dims),
    )  # (*, C, H, r)
    x = x.reshape(*batch_dims, c, h * r)  # (*, C, H*r)
    return x


class DCDownBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
        shortcut: bool = True,
    ) -> None:
        super().__init__()

        self.downsample = downsample
        self.factor = 2
        self.stride = 1 if downsample else 2
        self.group_size = in_channels * self.factor // out_channels
        self.shortcut = shortcut

        out_ratio = self.factor
        if downsample:
            assert out_channels % out_ratio == 0
            out_channels = out_channels // out_ratio

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.conv(hidden_states)
        if self.downsample:
            x = pixel_unshuffle(x, self.factor)

        if self.shortcut:
            y = pixel_unshuffle(hidden_states, self.factor)
            y = y.unflatten(1, (-1, self.group_size))
            y = y.mean(dim=2)
            hidden_states = x + y
        else:
            hidden_states = x
        return hidden_states


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch_norm",
        act_fn: str = "relu6",
    ) -> None:
        super().__init__()

        self.norm_type = norm_type

        self.nonlinearity = (
            get_activation(act_fn) if act_fn is not None else nn.Identity()
        )
        self.conv1 = nn.Conv1d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.norm = get_normalization(norm_type, out_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.norm_type == "rms_norm":
            # move channel to the last dimension so we apply RMSnorm across channel dimension
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states + residual


def get_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    attention_head_dim: int,
    norm_type: str,
    act_fn: str,
    qkv_mutliscales: Tuple[int] = (),
):
    if block_type == "ResBlock":
        block = ResBlock(in_channels, out_channels, norm_type, act_fn)

    else:
        raise ValueError(f"Block with {block_type=} is not supported.")

    return block


class DCEncoder1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        attention_head_dim: int = 32,
        block_type: Union[str, Tuple[str]] = "ResBlock",
        block_out_channels: Tuple[int] = (128, 256, 512, 512, 1024, 1024),
        layers_per_block: Tuple[int] = (2, 2, 2, 2, 2, 2),
        qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,)),
        downsample_block_type: str = "pixel_unshuffle",
        out_shortcut: bool = True,
    ):
        super().__init__()

        num_blocks = len(block_out_channels)

        if isinstance(block_type, str):
            block_type = (block_type,) * num_blocks

        if layers_per_block[0] > 0:
            self.conv_in = nn.Conv1d(
                in_channels,
                block_out_channels[0]
                if layers_per_block[0] > 0
                else block_out_channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv_in = DCDownBlock1d(
                in_channels=in_channels,
                out_channels=block_out_channels[0]
                if layers_per_block[0] > 0
                else block_out_channels[1],
                downsample=downsample_block_type == "pixel_unshuffle",
                shortcut=False,
            )

        down_blocks = []
        for i, (out_channel, num_layers) in enumerate(
            zip(block_out_channels, layers_per_block)
        ):
            down_block_list = []

            for _ in range(num_layers):
                block = get_block(
                    block_type[i],
                    out_channel,
                    out_channel,
                    attention_head_dim=attention_head_dim,
                    norm_type="rms_norm",
                    act_fn="silu",
                    qkv_mutliscales=qkv_multiscales[i],
                )
                down_block_list.append(block)

            if i < num_blocks - 1 and num_layers > 0:
                downsample_block = DCDownBlock1d(
                    in_channels=out_channel,
                    out_channels=block_out_channels[i + 1],
                    downsample=downsample_block_type == "pixel_unshuffle",
                    shortcut=True,
                )
                down_block_list.append(downsample_block)

            down_blocks.append(nn.Sequential(*down_block_list))

        self.down_blocks = nn.ModuleList(down_blocks)

        self.conv_out = nn.Conv1d(block_out_channels[-1], latent_channels, 3, 1, 1)

        self.out_shortcut = out_shortcut
        if out_shortcut:
            self.out_shortcut_average_group_size = (
                block_out_channels[-1] // latent_channels
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)

        if self.out_shortcut:
            x = hidden_states.unflatten(1, (-1, self.out_shortcut_average_group_size))
            x = x.mean(dim=2)
            hidden_states = self.conv_out(hidden_states) + x
        else:
            hidden_states = self.conv_out(hidden_states)

        return hidden_states


class DCUpBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        interpolate: bool = False,
        shortcut: bool = True,
        interpolation_mode: str = "nearest",
    ) -> None:
        super().__init__()

        self.interpolate = interpolate
        self.interpolation_mode = interpolation_mode
        self.shortcut = shortcut
        self.factor = 2
        self.repeats = out_channels * self.factor // in_channels

        out_ratio = self.factor

        if not interpolate:
            out_channels = out_channels * out_ratio

        self.conv = nn.Conv1d(in_channels, out_channels, 3, 1, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.interpolate:
            x = F.interpolate(
                hidden_states, scale_factor=self.factor, mode=self.interpolation_mode
            )
            x = self.conv(x)
        else:
            x = self.conv(hidden_states)
            x = pixel_shuffle(x, self.factor)

        if self.shortcut:
            y = hidden_states.repeat_interleave(self.repeats, dim=1)
            y = pixel_shuffle(y, self.factor)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class DCDecoder1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        attention_head_dim: int = 32,
        block_type: Union[str, Tuple[str]] = "ResBlock",
        block_out_channels: Tuple[int] = (128, 256, 512, 512, 1024, 1024),
        layers_per_block: Tuple[int] = (2, 2, 2, 2, 2, 2),
        qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,)),
        norm_type: Union[str, Tuple[str]] = "rms_norm",
        act_fn: Union[str, Tuple[str]] = "silu",
        upsample_block_type: str = "pixel_shuffle",
        in_shortcut: bool = True,
    ):
        super().__init__()

        num_blocks = len(block_out_channels)

        if isinstance(block_type, str):
            block_type = (block_type,) * num_blocks
        if isinstance(norm_type, str):
            norm_type = (norm_type,) * num_blocks
        if isinstance(act_fn, str):
            act_fn = (act_fn,) * num_blocks

        self.conv_in = nn.Conv1d(latent_channels, block_out_channels[-1], 3, 1, 1)

        self.in_shortcut = in_shortcut
        if in_shortcut:
            self.in_shortcut_repeats = block_out_channels[-1] // latent_channels

        up_blocks = []
        for i, (out_channel, num_layers) in reversed(
            list(enumerate(zip(block_out_channels, layers_per_block)))
        ):
            up_block_list = []

            if i < num_blocks - 1 and num_layers > 0:
                upsample_block = DCUpBlock1d(
                    block_out_channels[i + 1],
                    out_channel,
                    interpolate=upsample_block_type == "interpolate",
                    shortcut=True,
                )
                up_block_list.append(upsample_block)

            for _ in range(num_layers):
                block = get_block(
                    block_type[i],
                    out_channel,
                    out_channel,
                    attention_head_dim=attention_head_dim,
                    norm_type=norm_type[i],
                    act_fn=act_fn[i],
                    qkv_mutliscales=qkv_multiscales[i],
                )
                up_block_list.append(block)

            up_blocks.insert(0, nn.Sequential(*up_block_list))

        self.up_blocks = nn.ModuleList(up_blocks)

        channels = (
            block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1]
        )

        self.norm_out = RMSNorm(channels, 1e-5, elementwise_affine=True, bias=True)
        self.conv_act = nn.ReLU()
        self.conv_out = None

        if layers_per_block[0] > 0:
            self.conv_out = nn.Conv1d(channels, in_channels, 3, 1, 1)
        else:
            self.conv_out = DCUpBlock1d(
                channels,
                in_channels,
                interpolate=upsample_block_type == "interpolate",
                shortcut=False,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.in_shortcut:
            x = hidden_states.repeat_interleave(self.in_shortcut_repeats, dim=1)
            hidden_states = self.conv_in(hidden_states) + x
        else:
            hidden_states = self.conv_in(hidden_states)

        for up_block in reversed(self.up_blocks):
            hidden_states = up_block(hidden_states)

        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1

FSQ
-------------------------
Example of usage:

quantizer = FSQ(z_dim, fsq_levels)
z_float = Encoder(x)
z_quant, code_index = quantizer(z_float)

"""

# helper functions


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# tensor helpers


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


# main class


class FSQ(Module):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim)
            if has_projections
            else nn.Identity()
        )
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(
            torch.arange(self.codebook_size), project_out=False
        )
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(self, indices: Tensor, project_out=True) -> Tensor:
        """Inverse of `codes_to_indices`."""

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack_one(z, "b * d")

        assert z.shape[-1] == self.dim, (
            f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"
        )

        z = self.project_in(z)

        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, "b n c d -> b n (c d)")

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")

            indices = unpack_one(indices, ps, "b * c")

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        return out, indices


if __name__ == "__main__":
    levels = [8, 5, 5, 5]  # see 4.1 and A.4.1 in the paper
    quantizer = FSQ(levels)

    x = torch.randn(1, 4, 16, 16)  # 4 since there are 4 levels
    xhat, indices = quantizer(x)

    print(xhat.shape)  # (1, 4, 16, 16) - same as input dimensions
    print(indices.shape)  # (1, 16, 16)    - the 4 levels are converted to code index


def multiplyList(myList):
    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * x
    return result


class SurfaceFSQVAE(LightningModule):
    def __init__(
        self,
        lr: float = 5e-4,
        weight_decay: float = 1e-5,
        num_down_blocks: int = 4,
        num_up_blocks: int = 4,
        block_out_channels: Union[str, List[int]] = "128,256,512,512",
        layers_per_block: int = 2,
        latent_channels: int = 3,
        sync_dist_train: bool = True,
        fsq_levels: list = [8, 5, 5, 5],
        z_dim: int = 4,
        max_face: int = 0,  # Saving as a hyperparameter and copying over from datamodule args
        max_edge: int = 0,  # Saving as a hyperparameter and copying over from datamodule args
        use_dcae: bool = False,  # use deepcompression autorencoder
    ):
        """
        Initialize the SurfaceFSQVAE model.

        Parameters
        ----------
        lr : float
            Learning rate.
        weight_decay : float
            Weight decay in the AdamW optimizer.
        num_down_blocks : int
            Number of down blocks in the encoder that controls the latent grid spatial dimensions.
        num_up_blocks : int
            Number of up blocks in the decoder that controls the output grid spatial dimensions.
        block_out_channels : Union[str, List[int]]
            Number of output channels for each block in the encoder and decoder.
        layers_per_block : int
            Number of layers in each block.
        latent_channels : int
            Number of latent grid channels.
        sync_dist_train : bool
            Whether to synchronize distributed training.
        fsq_levels : list
            List of levels for the FSQ quantizer. The codebook size is the product of this.
        """
        super().__init__()
        self.save_hyperparameters()
        if isinstance(block_out_channels, str):
            self.hparams.block_out_channels = list(
                map(int, block_out_channels.split(","))
            )

        in_channels = 3  # We always use the point coordinate grids

        if use_dcae:
            self.encoder = DCEncoder(
                in_channels=in_channels,
                latent_channels=z_dim,
                block_out_channels=self.hparams.block_out_channels,
                layers_per_block=[layers_per_block] * num_up_blocks,
            )
        else:
            self.encoder = Encoder(
                in_channels=in_channels,
                out_channels=z_dim,
                down_block_types=["DownEncoderBlock2D"] * num_down_blocks,
                block_out_channels=self.hparams.block_out_channels,
                layers_per_block=layers_per_block,
                act_fn="silu",
                norm_num_groups=32,
                double_z=False,
            )

        # pass init params to Decoder
        out_channels = 3  # Points are always predicted

        if use_dcae:
            self.decoder = DCDecoder(
                in_channels=out_channels,
                latent_channels=z_dim,
                block_out_channels=self.hparams.block_out_channels,
                layers_per_block=[layers_per_block] * num_up_blocks,
                act_fn="silu",
            )
        else:
            self.decoder = Decoder(
                in_channels=z_dim,
                out_channels=out_channels,
                up_block_types=["UpDecoderBlock2D"] * num_up_blocks,
                block_out_channels=self.hparams.block_out_channels,
                layers_per_block=layers_per_block,
                norm_num_groups=32,
                act_fn="silu",
            )

        self.quantizer = FSQ(dim=z_dim, levels=fsq_levels)

        self.codebook_size = multiplyList(fsq_levels)

        self.downsample = torch.nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1
        )
        self.upsample = torch.nn.Linear(16, 64)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def prepare_input(self, batch):
        features = [batch["face_points_normalized"]]
        input_grid = torch.cat(features, dim=-1).permute(0, 3, 1, 2)
        return input_grid

    def prepare_output(self, decoded_output):
        output = {
            "face_points_normalized": decoded_output[:, :3],
        }
        return output

    def common_step(self, batch):
        face_uv = self.prepare_input(batch)
        # Encode
        logits = self.encoder(face_uv)

        # further shrink it down
        logits_downsample = self.downsample(logits)

        # FSQ
        quant, id = self.quantizer(logits_downsample)

        quant_upsample = self.upsample(quant.reshape(len(quant), -1)).reshape(
            len(quant), 4, 4, 4
        )

        # Decode
        dec = self.decoder(quant_upsample)
        output = {
            "face_uv": face_uv,
            "z": quant,
            "dec": dec,
            "id": id,
        }
        return output

    def common_step_and_loss(self, batch, stage: str):
        assert stage in ("train", "val", "test")
        output = self.common_step(batch)
        dec = output["dec"]
        dec = self.prepare_output(dec)

        points_mse_loss = torch.nn.functional.mse_loss(
            dec["face_points_normalized"],
            batch["face_points_normalized"].permute(0, 3, 1, 2),
        )
        total_loss = points_mse_loss

        loss = {
            f"{stage}/points_mse_loss": points_mse_loss,
        }

        loss[f"{stage}/loss"] = total_loss
        self.log_dict(
            loss,
            on_step=stage == "train",  # Log per-step for training
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.hparams.sync_dist_train,
        )
        return total_loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.common_step_and_loss(batch, "train")

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        output = self.common_step(batch)
        dec = output["dec"]
        dec = self.prepare_output(dec)

        points_mse_loss = torch.nn.functional.mse_loss(
            dec["face_points_normalized"],
            batch["face_points_normalized"].permute(0, 3, 1, 2),
        )
        total_loss = points_mse_loss

        loss = {
            "val/points_mse_loss": points_mse_loss,
            "val/loss": total_loss,
        }

        self.log_dict(
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.hparams.sync_dist_train,
        )
        return total_loss

    def drop_decoder(self):
        self.decoder = None
        return self

    def drop_encoder(self):
        self.encoder = None
        return self

    def encode(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            x (`torch.FloatTensor`): Input sample.
            posterior_sample_generator (`torch.Generator`, *optional*, defaults to None):
                The generator to use to sample the posterior. Returns the mode if undefined.
        """
        logits = self.encoder(x)
        logits_downsample = self.downsample(logits)
        quant, id = self.quantizer(logits_downsample)
        return quant, id

    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            z (`torch.FloatTensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        z_upsample = self.upsample(z.reshape(len(z), -1)).reshape(len(z), 4, 4, 4)
        dec = self.decoder(z_upsample)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)


class EdgeFSQVAE(LightningModule):
    def __init__(
        self,
        lr: float = 5e-4,
        weight_decay: float = 1e-5,
        num_down_blocks: int = 3,
        num_up_blocks: int = 3,
        block_out_channels: Union[str, List[int]] = "128,256,512",
        layers_per_block: int = 2,
        latent_channels: int = 3,
        sync_dist_train: bool = True,
        fsq_levels: list = [8, 5, 5, 5],
        z_dim: int = 4,
        max_face: int = 0,  # Saving as a hyperparameter and copying over from datamodule args
        max_edge: int = 0,  # Saving as a hyperparameter and copying over from datamodule args
        use_dcae: bool = False,  # use deepcompression autoencoder
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if isinstance(block_out_channels, str):
            self.hparams.block_out_channels = list(
                map(int, block_out_channels.split(","))
            )

        in_channels = 3  # TODO always use the point coordinate for edges
        if use_dcae:
            self.encoder = DCEncoder1D(
                in_channels=in_channels,
                latent_channels=z_dim,
                block_out_channels=self.hparams.block_out_channels,
                layers_per_block=[layers_per_block] * num_down_blocks,
            )
        else:
            self.encoder = Encoder1D(
                in_channels=in_channels,
                out_channels=z_dim,
                down_block_types=["DownBlock1D"] * num_down_blocks,
                block_out_channels=self.hparams.block_out_channels,
                layers_per_block=layers_per_block,
                act_fn="silu",
                norm_num_groups=32,
                double_z=False,
            )

        out_channels = 3  # TODO always use the point coordinate for edges
        if use_dcae:
            self.decoder = DCDecoder1D(
                in_channels=out_channels,
                latent_channels=z_dim,
                block_out_channels=self.hparams.block_out_channels,
                layers_per_block=[layers_per_block] * num_up_blocks,
                act_fn="silu",
            )
        else:
            self.decoder = Decoder1D(
                in_channels=z_dim,
                out_channels=out_channels,
                up_block_types=["UpBlock1D"] * num_up_blocks,
                block_out_channels=self.hparams.block_out_channels,
                layers_per_block=layers_per_block,
                act_fn="silu",
                norm_num_groups=32,
            )

        self.quantizer = FSQ(dim=z_dim, levels=fsq_levels)

        self.codebook_size = multiplyList(fsq_levels)
        self.downsample = torch.nn.Conv1d(
            in_channels=z_dim, out_channels=z_dim, kernel_size=3, stride=2, padding=1
        )
        self.upsample = torch.nn.Linear(2 * z_dim, 4 * z_dim)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def prepare_input(self, batch):
        features = [batch["edge_points_normalized"]]
        input_grid = torch.cat(features, dim=-1).permute(0, 2, 1)
        return input_grid

    def prepare_output(self, decoded_output):
        output = {
            "edge_points_normalized": decoded_output[:, :3],
        }
        return output

    def common_step(self, batch):
        edge_u = self.prepare_input(batch)
        # Encode
        z = self.encoder(edge_u)

        z_downsample = self.downsample(z)

        # FSQ
        quant, id = self.quantizer(z_downsample.permute(0, 2, 1))
        quant = quant.permute(0, 2, 1)

        quant_upsample = self.upsample(quant.reshape(len(quant), -1)).reshape(
            len(quant), self.hparams.z_dim, 4
        )

        # Decode
        dec = self.decoder(quant_upsample)

        return {
            "edge_u": edge_u,
            "z": quant,
            "dec": dec,
        }

    def common_step_and_loss(self, batch, stage: str):
        assert stage in ("train", "val", "test")
        output = self.common_step(batch)
        dec = self.prepare_output(output["dec"])

        # Point grid loss: first 3 channels are always point grids
        points_mse_loss = torch.nn.functional.mse_loss(
            dec["edge_points_normalized"],
            batch["edge_points_normalized"].permute(0, 2, 1),
        )
        total_loss = points_mse_loss

        loss = {
            f"{stage}/points_mse_loss": points_mse_loss,
        }

        loss[f"{stage}/loss"] = total_loss.item()
        self.log_dict(
            loss,
            on_step=stage == "train",  # Log per-step for training
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.hparams.sync_dist_train,
        )
        return total_loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.common_step_and_loss(batch, "train")

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.common_step_and_loss(batch, "val")

    def drop_decoder(self):
        self.decoder = None
        return self

    def drop_encoder(self):
        self.encoder = None
        return self

    def encode(
        self,
        edge_u: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            x (`torch.FloatTensor`): Input sample.
            posterior_sample_generator (`torch.Generator`, *optional*, defaults to None):
                The generator to use to sample the posterior. Returns the mode if undefined.
        """
        # Encode
        z = self.encoder(edge_u)
        z_downsample = self.downsample(z)

        # FSQ
        quant, id = self.quantizer(z_downsample.permute(0, 2, 1))
        quant = quant.permute(0, 2, 1)
        return quant, id

    def decode(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            z (`torch.FloatTensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        quant_upsample = self.upsample(x.reshape(len(x), -1)).reshape(
            len(x), self.hparams.z_dim, 4
        )
        dec = self.decoder(quant_upsample)
        return DecoderOutput(sample=dec)
