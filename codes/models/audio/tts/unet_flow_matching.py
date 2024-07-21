import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from trainer.networks import register_model

from models.diffusion.nn import normalization, zero_module
from models.diffusion.unet_diffusion import TimestepEmbedSequential, TimestepBlock, QKVAttentionLegacy
from models.lucidrains.x_transformers import RelativePositionBias
from models.diffusion.nn import timestep_embedding, normalization, zero_module, conv_nd, linear
from utils.util import checkpoint
import random
from torch import autocast  

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        do_checkpoint=True,
        relative_pos_embeddings=False,
    ):
        super().__init__()
        self.channels = channels
        self.do_checkpoint = do_checkpoint
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        # split heads before split qkv
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(scale=(channels // self.num_heads) ** .5, causal=False, heads=num_heads, num_buckets=32, max_distance=64)
        else:
            self.relative_pos_embeddings = None

    def forward(self, x, mask=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv, mask, self.relative_pos_embeddings)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)
    
class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        dims=2,
        kernel_size=3,
        efficient_config=True,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        padding = {1: 0, 3: 1, 5: 2}[kernel_size]
        eff_kernel = 1 if efficient_config else 3
        eff_padding = 0 if efficient_config else 1

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, eff_kernel, padding=eff_padding),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, eff_kernel, padding=eff_padding)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, x, emb
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h



# helper functions for adaptive layer norm
def modulate(x, shift, scale):
    return x * (x + scale[:, None] + shift[:, None])

class DiffusionLayer(TimestepBlock):
    def __init__(self, model_channels, dropout, num_heads):
        super().__init__()
        self.resblk = ResBlock(model_channels, model_channels, dropout, model_channels, dims=1, use_scale_shift_norm=True)
        self.attn = AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True)

    def forward(self, x, time_emb, mask=None):
        y = self.resblk(x, time_emb)
        return self.attn(y, mask)


# optimal transport conditional flow matching loss
class OTCFMLoss(nn.Module):
    def __init__(self):
        super(OTCFMLoss, self).__init__()

    def forward(self, mask_padding, ut, vt):
        # pdb.set_trace()
        n_elements = (mask_padding).sum()
        n_mel_dims = vt.shape[1]
        loss_otcfm = ((vt - ut) ** 2) * mask_padding.unsqueeze(1)
        loss_otcfm = torch.sum(loss_otcfm) / (n_elements * n_mel_dims)
        return loss_otcfm


# optimal transport conditional flow matching
class OTCFM(nn.Module):
    def __init__(self, 
                 n_hidden=1024, 
                 in_latent_channels=1024,
                 n_mel_channels=100, 
                 n_blocks=12,
                 n_heads=12, 
                 p_dropout=0.1, 
                 layer_drop=.1,
                 use_fp16=False,
                 sigma=0.0001
                 ):
        super(OTCFM, self).__init__()

        self.n_hidden = n_hidden
        self.n_mel_channels = n_mel_channels
        self.sigma = sigma
        self.enable_fp16 = use_fp16
        self.layer_drop = layer_drop
        self.inp_block = conv_nd(1, n_mel_channels, n_hidden, 3, 1, 1)
        self.time_embed = nn.Sequential(
            linear(n_hidden, n_hidden),
            nn.SiLU(),
            linear(n_hidden, n_hidden),
        )

        self.contextual_embedder = nn.Sequential(nn.Conv1d(n_mel_channels,n_hidden,3,padding=1,stride=2),
                                            nn.Conv1d(n_hidden, n_hidden*2,3,padding=1,stride=2),
                                            AttentionBlock(n_hidden*2, n_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                            AttentionBlock(n_hidden*2, n_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                            AttentionBlock(n_hidden*2, n_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                            AttentionBlock(n_hidden*2, n_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                            AttentionBlock(n_hidden*2, n_heads, relative_pos_embeddings=True, do_checkpoint=False))

        self.latent_conditioner = nn.Sequential(
            nn.Conv1d(in_latent_channels, n_hidden, 3, padding=1),
            AttentionBlock(n_hidden, n_heads, relative_pos_embeddings=True),
            AttentionBlock(n_hidden, n_heads, relative_pos_embeddings=True),
            AttentionBlock(n_hidden, n_heads, relative_pos_embeddings=True),
            AttentionBlock(n_hidden, n_heads, relative_pos_embeddings=True),
        )

        self.conditioning_timestep_integrator = TimestepEmbedSequential(
            DiffusionLayer(n_hidden, p_dropout, n_heads),
            DiffusionLayer(n_hidden, p_dropout, n_heads),
            DiffusionLayer(n_hidden, p_dropout, n_heads),
        )

        self.code_norm = normalization(n_hidden)
        self.integrating_conv = nn.Conv1d(n_hidden*2, n_hidden, kernel_size=1)

        self.layers = nn.ModuleList([DiffusionLayer(n_hidden, p_dropout, n_heads) for _ in range(n_blocks)] +
                                    [ResBlock(n_hidden, n_hidden, p_dropout, dims=1, use_scale_shift_norm=True) for _ in range(3)])

        self.out = nn.Sequential(
            normalization(n_hidden),
            nn.SiLU(),
            zero_module(conv_nd(1, n_hidden, n_mel_channels, 3, padding=1)),
        )

        self.loss_otcfm = OTCFMLoss()

    def get_cond_emb(self, latent, conditioning_input, expected_seq_len):
        latent = latent.permute(0, 2, 1)
        # Note: this block does not need to repeated on inference, since it is not timestep-dependent or x-dependent.
        speech_conditioning_input = conditioning_input.unsqueeze(1) if len(
            conditioning_input.shape) == 3 else conditioning_input
        conds = []
        for j in range(speech_conditioning_input.shape[1]):
            conds.append(self.contextual_embedder(speech_conditioning_input[:, j]))
        conds = torch.cat(conds, dim=-1)
        cond_emb = conds.mean(dim=-1)
        cond_scale, cond_shift = torch.chunk(cond_emb, 2, dim=1)

        code_emb = self.latent_conditioner(latent)

        code_emb = self.code_norm(code_emb) * (1 + cond_scale.unsqueeze(-1)) + cond_shift.unsqueeze(-1)

        expanded_code_emb = F.interpolate(code_emb, size=expected_seq_len, mode='nearest')

        return expanded_code_emb


    def forward(self, x1, latent, conditioning_input, wav_lens, in_sample_rate=22050, out_sample_rate=24000, hop_length=256):

        mel_lens = (wav_lens * (out_sample_rate/in_sample_rate) / hop_length).int()

        cond = self.get_cond_emb(latent, conditioning_input, x1.shape[-1])

        mask = torch.arange(cond.shape[-1], device=cond.device).unsqueeze(0) < mel_lens.unsqueeze(1)

        B, D, T = x1.shape

        t = torch.rand(
            (B, 1, 1), dtype=x1.dtype, device=x1.device, requires_grad=False)

        # spherical gaussian noise
        x0 = torch.randn_like(x1)

        # interpolation between t-scaled x0 and x1
        xt = (1 - (1 - self.sigma) * t) * x0 + t * x1

        # interpolation between x1 and sigma scaled x0
        ut = x1 - x0 * (1 - self.sigma)

        # mask cond at random for classifier free guidance
        mask_cfg = np.random.random(B) < 0.1
        cond[mask_cfg] *= 0

        time_emb = self.time_embed(timestep_embedding(t[:, 0, 0], self.n_hidden, scale=1000))

        code_emb = self.conditioning_timestep_integrator(cond, time_emb)

        x = self.inp_block(xt)
        x = torch.cat([x, code_emb], dim=1)
        x = self.integrating_conv(x)

        unused_params = []
        for i, lyr in enumerate(self.layers):
            if self.training and self.layer_drop > 0 and i != 0 and i != (len(self.layers)-1) and random.random() < self.layer_drop:
                unused_params.extend(list(lyr.parameters()))
            else:
                # First and last blocks will have autocast disabled for improved precision.
                with autocast(x.device.type, enabled=self.enable_fp16 and i != 0):
                    x = lyr(x, time_emb)

        x = x.float()
        vt = self.out(x)

        loss_otcfm = self.loss_otcfm(mask, ut, vt)

        return loss_otcfm

@register_model
def register_otcfm(opt_net, opt):
    return OTCFM(**opt_net['kwargs'])