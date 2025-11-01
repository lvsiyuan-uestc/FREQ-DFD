# model/freq_branch.py
import torch
import torch.nn as nn
from typing import Tuple
from utils.freq import block_dct_maps


class DCTBranch(nn.Module):
    """
    频域分支：
    - 输入 : x_rgb ∈ [0,1], 形状 (B,3,H,W)
    - 处理 : block_dct_maps -> (B,K,H/8,W/8)，随后 3×Conv 下采样 + GAP
    - 输出 : f_freq ∈ R^{B×out_dim}，默认 out_dim=1024（训练脚本里会用 visual_hidden_size 覆盖）
    - dtype : 全流程使用 float32，避免 AMP 下的 BF16/FP32 类型冲突
    """
    def __init__(self, k_keep: int = 16, hidden: int = 64, out_dim: int = 1280):
        super().__init__()
        self.k_keep = k_keep

        # 卷积分支（float32）
        self.net = nn.Sequential(
            nn.Conv2d(k_keep, hidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),  # /4
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> B×hidden×1×1
        )
        self.proj = nn.Linear(hidden, out_dim)

        # 明确设为 float32，避免和外部 AMP/BF16 混用引发类型错配
        self.float()

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """
        x_rgb: (B,3,H,W) in [0,1]
        return: (B, out_dim) float32
        """
        # block_dct_maps 内部做 8×8 DCT 与子带选择，返回 (B,K,H/8,W/8)，确保 float32
        dct_map = block_dct_maps(x_rgb.float(), k_keep=self.k_keep)    # (B,K,H/8,W/8) float32
        h = self.net(dct_map).flatten(1)                               # (B, hidden) float32
        f_freq = self.proj(h).float().contiguous()                     # (B, out_dim) float32
        return f_freq


class GatedFuse(nn.Module):
    """
    门控融合：
    - 输入 : f_rgb, f_freq 支持 (B,D) 或 (B,T,D)
    - 若 token 维 T 不一致，自动广播或做均值池化对齐
    - 输出 : 与输入对齐 (B,D) 或 (B,T,D)
    - 初始化偏置为 -2.0，使初期更偏向频域特征（sigmoid≈0.12）
    - dtype : 计算在 float32
    """
    def __init__(self, dim: int = 1280, hidden: int = 256, return_gate: bool = False):
        super().__init__()
        self.return_gate = return_gate
        self.ln = nn.LayerNorm(dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.constant_(self.mlp[-1].bias, -2.0)
        self.float()  # 显式 float32

    def forward(self, f_rgb: torch.Tensor, f_freq: torch.Tensor):
        """
        f_rgb : (B,D) or (B,T,D)
        f_freq: (B,D) or (B,T,D)
        return: same shape as inputs (B,D) or (B,T,D), float32
        """
        f_rgb = f_rgb.float().contiguous()
        f_freq = f_freq.float().contiguous()

        squeeze_back = False
        if f_rgb.dim() == 2:
            f_rgb = f_rgb.unsqueeze(1)      # (B,1,D)
            squeeze_back = True
        if f_freq.dim() == 2:
            f_freq = f_freq.unsqueeze(1)    # (B,1,D)

        # 对齐 token 维
        Tr, Tf = f_rgb.size(1), f_freq.size(1)
        if Tr != Tf:
            if Tr == 1 and Tf > 1:
                f_rgb = f_rgb.expand(-1, Tf, -1)
            elif Tf == 1 and Tr > 1:
                f_freq = f_freq.expand(-1, Tr, -1)
            else:
                f_rgb  = f_rgb.mean(dim=1, keepdim=True)
                f_freq = f_freq.mean(dim=1, keepdim=True)

        h = torch.cat([f_rgb, f_freq], dim=-1)  # (B,T,2D)
        h = self.ln(h)
        g = torch.sigmoid(self.mlp(h))          # (B,T,1), float32

        out = g * f_rgb + (1.0 - g) * f_freq    # (B,T,D) float32

        if squeeze_back:
            out = out.squeeze(1)                # (B,D)
            g = g.squeeze(1)                    # (B,1) -> (B,)

        return (out, g) if self.return_gate else out
