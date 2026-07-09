import math

import torch.nn as nn

from rfsensing.models import register


@register("lstm")
class LSTM(nn.Module):
    """LSTM over one axis of the sample treated as time.

    The sample is permuted so ``seq_axis`` comes first, the remaining axes
    are flattened into per-step features, and the last hidden state is
    classified. ``bidirectional=True`` gives a BiLSTM.
    """

    def __init__(
        self,
        in_shape,
        num_classes,
        hidden_size=128,
        num_layers=1,
        bidirectional=False,
        seq_axis=-1,
        dropout=0.0,
    ):
        super().__init__()
        self.seq_axis = seq_axis % len(in_shape)
        feat_dim = math.prod(in_shape) // in_shape[self.seq_axis]
        self.lstm = nn.LSTM(
            feat_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        x = x.movedim(self.seq_axis + 1, 1)  # +1 for the batch dim -> (B, T, ...)
        x = x.flatten(start_dim=2)  # (B, T, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1])
