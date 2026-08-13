"""
Model factory for the CGMTransformer federated template.

Dispatches on cfg.model_name:
  - "transformer"        : original per-timestep tokenisation (CGMTransformerModel)
  - "transformer_patch"  : non-overlapping patched encoder    (TransformerPatchModel)

Default is "transformer" so existing configs without model_name keep working.
"""

from .model.transformer_model import CGMTransformerModel
from .model.transformer_patch_model import TransformerPatchModel


_BUILDERS = {
    "transformer":       CGMTransformerModel,
    "transformer_patch": TransformerPatchModel,
}


def build_transformer(cfg):
    """Instantiate the configured CGMTransformer variant.

    Expected cfg attributes vary per variant; see the model classes for the
    full list.  Common: enc_in, dec_in, c_out, d_model, n_heads, e_layers,
    d_layers, d_ff, dropout, embed, freq, activation, factor, full_attention,
    output_attention, pred_len.  transformer_patch additionally needs seq_len
    and patch_size.
    """
    name = str(getattr(cfg, 'model_name', 'transformer')).lower()
    try:
        cls = _BUILDERS[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown model_name='{name}'. "
            f"Available: {sorted(_BUILDERS)}"
        ) from e
    return cls(cfg)
