"""WhoFi signal encoders (Avola et al. 2025, arXiv:2507.12869).

WhoFi treats a CSI sample as a time sequence of flattened
antenna x subcarrier feature vectors, encodes it with an LSTM, BiLSTM, or
Transformer encoder, and projects the pooled representation through a linear
"signature module" onto an embedding matched by cosine similarity.

Faithful-to-paper choices: sinusoidal (non-trainable) positional encoding,
one Transformer layer by default (their best configuration), last-hidden-state
readout for the recurrent encoders, and sequences subsampled to ~100 packets
(their best packet length). Choices the paper leaves unspecified — model
width, head count, feed-forward width, signature dimension, and how the
Transformer output is pooled — are set to reasonable defaults here (mean
pooling over time) and exposed as keyword arguments.
"""

import math

import torch
import torch.nn as nn

from rfsensing.models import register


def _sinusoidal_encoding(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros(length, dim)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term[: dim // 2])
    return encoding


@register("whofi")
class WhoFi(nn.Module):
    """WhoFi sequence encoder with a linear signature head.

    ``encoder`` selects ``"transformer"`` (default), ``"lstm"``, or
    ``"bilstm"``. The sample axis ``seq_axis`` is treated as time and the
    remaining axes are flattened into per-step features; the sequence is
    uniformly subsampled to ``seq_len`` steps (``None`` keeps every step).
    """

    def __init__(
        self,
        in_shape,
        num_classes,
        *,
        encoder="transformer",
        seq_len=100,
        seq_axis=-1,
        d_model=128,
        num_layers=1,
        num_heads=8,
        ff_dim=256,
        dropout=0.1,
        embed_dim=128,
    ):
        super().__init__()
        if encoder not in ("transformer", "lstm", "bilstm"):
            raise ValueError(
                "encoder must be 'transformer', 'lstm', or 'bilstm', "
                f"got {encoder!r}"
            )
        self.seq_axis = seq_axis % len(in_shape)
        steps = in_shape[self.seq_axis]
        if seq_len is not None and not 1 <= seq_len <= steps:
            raise ValueError(
                f"seq_len must be in [1, {steps}], got {seq_len}"
            )
        self.encoder_type = encoder
        feat_dim = math.prod(in_shape) // steps
        length = seq_len or steps
        if seq_len is not None:
            indices = torch.linspace(0, steps - 1, seq_len).round().long()
            self.register_buffer("step_indices", indices, persistent=False)
        else:
            self.step_indices = None

        if encoder == "transformer":
            self.input_proj = nn.Linear(feat_dim, d_model)
            self.register_buffer(
                "positional_encoding",
                _sinusoidal_encoding(length, d_model),
                persistent=False,
            )
            layer = nn.TransformerEncoderLayer(
                d_model,
                num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            encoded_dim = d_model
        else:
            bidirectional = encoder == "bilstm"
            self.encoder = nn.LSTM(
                feat_dim,
                d_model,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            encoded_dim = d_model * (2 if bidirectional else 1)
        self.signature = nn.Linear(encoded_dim, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def _sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = x.movedim(self.seq_axis + 1, 1)  # +1 for the batch dim
        x = x.flatten(start_dim=2)  # (B, T, F)
        if self.step_indices is not None:
            x = x[:, self.step_indices]
        return x

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self._sequence(x)
        if self.encoder_type == "transformer":
            x = self.input_proj(x) + self.positional_encoding
            encoded = self.encoder(x).mean(dim=1)
        else:
            output, _ = self.encoder(x)
            if self.encoder_type == "bilstm":
                hidden = self.encoder.hidden_size
                # Forward direction ends at the last step, backward at the
                # first; concatenating both matches the paper's readout.
                encoded = torch.cat(
                    [output[:, -1, :hidden], output[:, 0, hidden:]], dim=1
                )
            else:
                encoded = output[:, -1]
        return self.signature(encoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))
