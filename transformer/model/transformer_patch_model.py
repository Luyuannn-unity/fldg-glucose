"""
CGMTransformer with patched encoder embedding.

Identical to CGMTransformerModel except the encoder input is split into
non-overlapping patches of `patch_size` timesteps along the time
dimension, each linearly projected to d_model.  The decoder is
unchanged (per-timestep tokens of length label_len + pred_len).

Forward signature is identical to CGMTransformerModel, so this is a drop-in
replacement at the call sites in flock_model_transformer.py.
"""

import torch
import torch.nn as nn

from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from .layers.Embed import DataEmbedding, PositionalEmbedding
from .layers.SelfAttention_Family import FullAttention, AttentionLayer, SparseAttention


# Mirrors the freq → mark-dim table in layers/Embed.py::TimeFeatureEmbedding.
_FREQ_TO_MARK_DIM = {
    'h': 4, 't': 3, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3,
    'cyc': 2, 'cyc3': 3,
}


class PatchEmbedding(nn.Module):
    """Non-overlapping patch embedding for time series.

    Input  : x      (B, L, c_in), x_mark (B, L, d_mark)
    Output :        (B, L // patch_size, d_model)

    Marks are mean-pooled within each patch, then linearly projected.
    """

    def __init__(self, c_in, d_model, patch_size, n_patches,
                 freq='cyc', dropout=0.1):
        super().__init__()
        self.patch_size = int(patch_size)
        self.n_patches  = int(n_patches)

        self.value_proj = nn.Linear(self.patch_size * c_in, d_model, bias=False)

        # Sinusoidal positional encoding indexed by patch position.
        self.position_embedding = PositionalEmbedding(
            d_model=d_model, max_len=max(5000, self.n_patches)
        )

        d_mark = _FREQ_TO_MARK_DIM[freq]
        self.temporal_embedding = nn.Linear(d_mark, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        B, L, C = x.shape
        P = self.patch_size
        if L % P != 0:
            raise ValueError(
                f"PatchEmbedding: seq_len={L} not divisible by patch_size={P}"
            )
        N = L // P

        # (B, L, C) -> (B, N, P, C) -> (B, N, P*C) -> (B, N, d_model)
        x = x.reshape(B, N, P, C).reshape(B, N, P * C)
        x = self.value_proj(x)

        # Mean-pool marks within each patch, then project.
        Bm, Lm, Dm = x_mark.shape
        marks = x_mark.reshape(Bm, N, P, Dm).mean(dim=2)
        marks = self.temporal_embedding(marks)

        x = x + marks + self.position_embedding(x)
        return self.dropout(x)


class TransformerPatchModel(nn.Module):
    """CGMTransformer variant with non-overlapping patched encoder input.

    Encoder operates on seq_len // patch_size tokens, each of width
    d_model (the "token size").  Decoder is unchanged — same per-timestep
    embedding and same projection to c_out quantiles.

    Required config attributes:
      enc_in, dec_in, c_out, d_model, n_heads, e_layers, d_layers, d_ff,
      dropout, embed, freq, activation, factor, full_attention,
      output_attention, pred_len, seq_len, patch_size
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.output_attention = getattr(configs, 'output_attention', False)
        self.patch_size = int(getattr(configs, 'patch_size', 6))

        seq_len = int(getattr(configs, 'seq_len', 72))
        if seq_len % self.patch_size != 0:
            raise ValueError(
                f"TransformerPatchModel: seq_len={seq_len} must be divisible "
                f"by patch_size={self.patch_size}"
            )
        self.n_patches = seq_len // self.patch_size

        self.enc_embedding = PatchEmbedding(
            c_in=configs.enc_in,
            d_model=configs.d_model,
            patch_size=self.patch_size,
            n_patches=self.n_patches,
            freq=configs.freq,
            dropout=configs.dropout,
        )

        # Decoder embedding unchanged — per-timestep tokens.
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        attn_cls = FullAttention if getattr(configs, 'full_attention', True) else SparseAttention

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        attn_cls(False, configs.factor,
                                 attention_dropout=configs.dropout,
                                 output_attention=self.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        attn_cls(True, configs.factor,
                                 attention_dropout=configs.dropout,
                                 output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        attn_cls(False, configs.factor,
                                 attention_dropout=configs.dropout,
                                 output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]
