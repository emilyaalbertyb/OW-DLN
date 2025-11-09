import torch
import torch.nn as nn
import torch.nn.functional as F


class DAAM(nn.Module):
    """
    Defect-Aware Attention Module (DAAM)
    - Sequential design: Channel -> Spatial
    - Defect guidance: integrates DVPG mask M_DGM into spatial attention
    - Three ablation switches:
        use_channel: enable/disable channel attention
        use_spatial: enable/disable spatial attention
        use_defect_guidance: enable/disable defect-guided mask

    Usage:
        y = DAAM(...)(x, mask)             # pass mask (B,1,H,W) directly in forward
        or
        daam.set_mask(mask); y = daam(x)   # cache the mask before forward

    Input shapes:
        x:    (B, C, H, W)
        mask: (B, 1, Hm, Wm)  -> automatically resized to (H, W)
    """

    def __init__(
        self,
        channels: int,
        spatial_kernel: int = 7,
        reduction: int = 16,
        lambda_guidance: float = 0.1,
        use_channel: bool = True,
        use_spatial: bool = True,
        use_defect_guidance: bool = True,
    ):
        super().__init__()
        assert spatial_kernel in (3, 7), "spatial_kernel should be 3 or 7 (CBAM default is 7)"
        self.C = channels
        self.lambda_guidance = lambda_guidance
        self.use_channel = use_channel
        self.use_spatial = use_spatial
        self.use_defect_guidance = use_defect_guidance

        # -------- Channel Attention --------
        if use_channel:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            hidden = max(channels // reduction, 1)
            self.mlp = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            )
            self.sigmoid_c = nn.Sigmoid()

        # -------- Spatial Attention --------
        if use_spatial:
            self.conv_spatial = nn.Conv2d(
                in_channels=2, out_channels=1,
                kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False
            )
            self.sigmoid_s = nn.Sigmoid()

        # Optional cached mask
        self._cached_mask = None

    def set_mask(self, mask: torch.Tensor):
        """
        Cache a defect mask for later usage.
        mask: (B, 1, H, W) with values in [0,1]
        """
        self._cached_mask = mask

    @staticmethod
    def _resize_like(m: torch.Tensor, x: torch.Tensor):
        return F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)

    @staticmethod
    def _minmax_normalize(t: torch.Tensor, eps: float = 1e-6):
        t_min = t.amin(dim=(-2, -1), keepdim=True)
        t_max = t.amax(dim=(-2, -1), keepdim=True)
        return (t - t_min) / (t_max - t_min + eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass
        x:    (B, C, H, W)
        mask: (B, 1, Hm, Wm) or None
        """
        assert x.dim() == 4, "Input must be (B, C, H, W)"
        B, C, H, W = x.shape
        out = x

        # ---------- Channel Attention ----------
        if self.use_channel:
            avg = self.avg_pool(out)     # (B, C, 1, 1)
            mx  = self.max_pool(out)     # (B, C, 1, 1)
            mc = self.mlp(avg) + self.mlp(mx)
            w_c = self.sigmoid_c(mc)
            out = out * w_c

        # ---------- Spatial Attention ----------
        if self.use_spatial:
            max_map, _ = out.max(dim=1, keepdim=True)         # (B,1,H,W)
            avg_map = out.mean(dim=1, keepdim=True)           # (B,1,H,W)
            sa_in = torch.cat([max_map, avg_map], dim=1)      # (B,2,H,W)
            w_s = self.sigmoid_s(self.conv_spatial(sa_in))    # (B,1,H,W)

            if self.use_defect_guidance:
                m = mask if mask is not None else self._cached_mask
                if m is not None:
                    if m.shape[-2:] != (H, W):
                        m = self._resize_like(m, out)
                    m = self._minmax_normalize(m)
                    w_s = w_s + self.lambda_guidance * m

            out = out * w_s

        return out
