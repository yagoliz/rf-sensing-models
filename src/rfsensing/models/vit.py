import torch
import torch.nn as nn

from rfsensing.models import register


@register("vit")
class ViT(nn.Module):
    """Small Vision Transformer over 2D CSI frames.

    Patches are cut with a strided Conv2d; rows/columns that do not fill a
    whole patch are dropped.
    """

    def __init__(
        self,
        in_shape,
        num_classes,
        patch_size=10,
        embed_dim=64,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
    ):
        super().__init__()
        channels, height, width = in_shape
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if height < patch_size[0] or width < patch_size[1]:
            raise ValueError(
                f"patch_size {patch_size} larger than input {(height, width)}"
            )
        num_patches = (height // patch_size[0]) * (width // patch_size[1])

        self.patch_embed = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim,
            num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def embed(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        return self.norm(self.encoder(x)[:, 0])

    def forward(self, x):
        return self.head(self.embed(x))