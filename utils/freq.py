# utils/freq.py
import torch
import torch.nn.functional as F

def _dct_mat(n=8, device="cpu", dtype=torch.float32):
    k = torch.arange(n, device=device, dtype=dtype).view(-1, 1)
    i = torch.arange(n, device=device, dtype=dtype).view(1, -1)
    alpha = torch.ones(n, device=device, dtype=dtype)
    alpha[0] = 1.0 / torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
    D = torch.sqrt(2.0 / torch.tensor(float(n), device=device, dtype=dtype)) * alpha.view(-1, 1) * torch.cos(
        (torch.pi * (2 * i + 1) * k) / (2.0 * n)
    )
    return D  # (n,n)

def block_dct_maps(x_rgb: torch.Tensor, k_keep: int = 16) -> torch.Tensor:
    """
    输入:  x_rgb (B,3,H,W) in [0,1]
    输出:  maps (B, K, H/8, W/8) 其中 K=k_keep (默认保留左上 sqrt(K)×sqrt(K) 低频)
    """
    assert x_rgb.dim() == 4 and x_rgb.size(1) == 3, "x must be (B,3,H,W)"
    B, C, H, W = x_rgb.shape
    assert H % 8 == 0 and W % 8 == 0, "H,W 应可整除 8"

    with torch.autocast(device_type="cuda", enabled=False):
        x = x_rgb.float()
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b  # (B,1,H,W)

        unfold = torch.nn.Unfold(kernel_size=8, stride=8)
        patches = unfold(y)                                  # (B,64,L)
        L = patches.size(-1)
        patches = patches.transpose(1, 2).contiguous()       # (B,L,64)
        patches = patches.view(B * L, 8, 8)                  # (B*L,8,8)

        D = _dct_mat(8, device=patches.device, dtype=patches.dtype)  # (8,8)
        tmp = torch.matmul(D, patches)
        dct = torch.matmul(tmp, D.t())                       # (B*L,8,8)

        s = int(k_keep ** 0.5)
        s = max(1, min(s, 8))
        low = dct[:, :s, :s]                                 # (B*L,s,s)

        K = s * s
        # ===== FIX: 用 reshape（或 .contiguous().view）避免 view 的连续性报错 =====
        low = low.reshape(B, L, K).permute(0, 2, 1).contiguous()   # (B,K,L)
        # =====================================================================

        h8, w8 = H // 8, W // 8
        maps = low.view(B, K, h8, w8)                       # (B,K,h8,w8)

        mean = maps.mean(dim=(0, 2, 3), keepdim=True)
        std  = maps.std(dim=(0, 2, 3), keepdim=True) + 1e-6
        maps = (maps - mean) / std

        return maps  # float32
