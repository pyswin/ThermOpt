"""SAU-FNO: U-FNO + Axial Self-Attention.

Inserts a pre-norm axial multi-head self-attention block after the LAST
U-Fourier layer (layer 5, conv5+w5+unet5) in U-FNO's SimpleBlock3d, BEFORE the
permute + fc1/fc2 projection. Full spatial MHSA is infeasible (64x64x8 = 32768
tokens -> ~343GB attention matrix), so attention is applied axially along Z, Y,
X separately (<=64 tokens/axis): cheap, captures long-range spatial deps, suits
the anisotropic structure of thermal fields.

Reuses SpectralConv3d / U_net / SimpleBlock3d from ufno.py (subclassed, so all
Fourier/U-Net layer params are inherited unchanged -> SAU-FNO differs from
U-FNO ONLY by the added attention block -> fair comparison)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from functools import reduce

from ufno import SimpleBlock3d  # set sys.path to include /home/menglinghai/ufno in the train script


class AxialAttentionBlock(nn.Module):
    """Pre-norm transformer block with axial multi-head self-attention (Z, Y, X)
    followed by an FFN. All sub-layers are residual.
    Input/output: (B, C, X, Y, Z)."""

    def __init__(self, channels, heads=4, ff_ratio=4):
        super().__init__()
        self.C = channels
        self.attn_z = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.attn_y = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.attn_x = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm_z = nn.LayerNorm(channels)
        self.norm_y = nn.LayerNorm(channels)
        self.norm_x = nn.LayerNorm(channels)
        self.norm_ff = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, ff_ratio * channels),
            nn.GELU(),
            nn.Linear(ff_ratio * channels, channels),
        )

    @staticmethod
    def _axial(x, attn, norm, axis):
        """Attend along one axis. axis: 0=X, 1=Y, 2=Z. Returns attn(norm(x)) in
        the original (B,C,X,Y,Z) layout (the residual delta)."""
        B, C, X, Y, Z = x.shape
        if axis == 2:  # Z-axis: seq=Z, batch=(B,X,Y)
            t = x.permute(0, 2, 3, 4, 1).reshape(B * X * Y, Z, C)
            t = norm(t)
            t = attn(t, t, t, need_weights=False)[0]
            return t.reshape(B, X, Y, Z, C).permute(0, 4, 1, 2, 3)
        elif axis == 1:  # Y-axis: seq=Y, batch=(B,X,Z)
            t = x.permute(0, 2, 4, 3, 1).reshape(B * X * Z, Y, C)
            t = norm(t)
            t = attn(t, t, t, need_weights=False)[0]
            return t.reshape(B, X, Z, Y, C).permute(0, 4, 1, 3, 2)
        else:           # X-axis: seq=X, batch=(B,Y,Z)
            t = x.permute(0, 3, 4, 2, 1).reshape(B * Y * Z, X, C)
            t = norm(t)
            t = attn(t, t, t, need_weights=False)[0]
            return t.reshape(B, Y, Z, X, C).permute(0, 4, 3, 1, 2)

    def forward(self, x):
        x = x + self._axial(x, self.attn_z, self.norm_z, 2)   # Z (len 8)
        x = x + self._axial(x, self.attn_y, self.norm_y, 1)   # Y (len 64)
        x = x + self._axial(x, self.attn_x, self.norm_x, 0)   # X (len 64)
        # FFN over channel dim (pre-norm, residual)
        B, C, X, Y, Z = x.shape
        t = x.permute(0, 2, 3, 4, 1).reshape(-1, C)
        t = self.ff(self.norm_ff(t))
        x = x + t.reshape(B, X, Y, Z, C).permute(0, 4, 1, 2, 3)
        return x


class SAUSimpleBlock3d(SimpleBlock3d):
    """U-FNO SimpleBlock3d + one AxialAttentionBlock inserted after the last
    U-Fourier layer (before the fc projection). All Fourier/U-Net layers are
    inherited from SimpleBlock3d unchanged."""

    def __init__(self, modes1, modes2, modes3, width, in_channels=12, attn_heads=4):
        super().__init__(modes1, modes2, modes3, width, in_channels=in_channels)
        self.axial_block = AxialAttentionBlock(self.width, heads=attn_heads)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = F.relu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = F.relu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = F.relu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet3(x)
        x = F.relu(x1 + x2 + x3)

        x1 = self.conv4(x)
        x2 = self.w4(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet4(x)
        x = F.relu(x1 + x2 + x3)

        x1 = self.conv5(x)
        x2 = self.w5(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet5(x)
        x = F.relu(x1 + x2 + x3)

        # ---- SAU-FNO: axial self-attention after the last U-Fourier layer ----
        x = self.axial_block(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class SAUNet3d(nn.Module):
    """Same pad-to-multiple-of-8 + crop wrapper as ufno.Net3d, but using
    SAUSimpleBlock3d (with the axial attention block)."""

    def __init__(self, modes1, modes2, modes3, width, in_channels=12, attn_heads=4):
        super(SAUNet3d, self).__init__()
        self.conv1 = SAUSimpleBlock3d(modes1, modes2, modes3, width,
                                      in_channels=in_channels, attn_heads=attn_heads)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        px = (8 - size_x % 8) % 8
        py = (8 - size_y % 8) % 8
        pz = (8 - size_z % 8) % 8
        x = F.pad(x, (0, 0, 0, pz, 0, py), "replicate")
        if px:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, px), 'constant', 0)
        x = self.conv1(x)
        x = x.view(batchsize, size_x + px, size_y + py, size_z + pz, 1)
        x = x[:, :size_x, :size_y, :size_z, :]
        return x.squeeze(-1)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c
