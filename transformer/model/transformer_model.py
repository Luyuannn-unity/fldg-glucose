"""
Self-contained CGMTransformer model for federated blood glucose prediction.

Adapted from bg_try_models/CGMTransformer-master/models/CGMTransformer.py.
Layer imports changed to relative so the model works without the original
project on sys.path.

Input/output shapes:
  x_enc      : (B, seq_len=72, 1)         – CGM encoder input
  x_mark_enc : (B, seq_len=72, 2)         – sin/cos time-of-day marks
  x_dec      : (B, label_len+pred_len, 1) – decoder input (context + zeros)
  x_mark_dec : (B, label_len+pred_len, 2) – time marks for decoder
  output     : (B, pred_len=24, c_out=5)  – quantile predictions
"""

import torch
import torch.nn as nn

from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from .layers.Embed import DataEmbedding
from .layers.SelfAttention_Family import FullAttention, AttentionLayer, SparseAttention


class CGMTransformerModel(nn.Module):
    """
    CGMTransformer: Gaussian-Process-inspired Transformer for CGM time-series forecasting.
    Wraps the original CGMTransformer.Model with self-contained layer imports.
    """

    def __init__(self, configs):
        super(CGMTransformerModel, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = getattr(configs, 'output_attention', False)

        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
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
        else:
            return dec_out[:, -self.pred_len:, :]
