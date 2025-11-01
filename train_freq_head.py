# train_freq_head.py
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageFile
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from model.openllama import OpenLLAMAPEFTModel
from model.freq_branch import DCTBranch, GatedFuse

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------
# Dataset
# -----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class FrameDataset(Dataset):
    def __init__(self, real_roots, fake_roots, size=224):
        self.items = []
        if isinstance(real_roots, (str, Path)): real_roots = [real_roots]
        if isinstance(fake_roots, (str, Path)): fake_roots = [fake_roots]
        for root in real_roots:
            r = Path(root)
            for p in sorted(r.iterdir()):
                if p.is_dir():
                    for x in sorted(p.iterdir()):
                        if x.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                            self.items.append((str(x), 0))
        for root in fake_roots:
            r = Path(root)
            for p in sorted(r.iterdir()):
                if p.is_dir():
                    for x in sorted(p.iterdir()):
                        if x.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                            self.items.append((str(x), 1))
        self.size = size


    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        im = Image.open(path).convert("RGB").resize((self.size, self.size))
        arr = np.asarray(im, dtype="float32") / 255.0  # [0,1]
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)
        return t, torch.tensor(y, dtype=torch.float32)

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_balanced_sampler(labels: List[int]) -> Tuple[WeightedRandomSampler, np.ndarray]:
    labels_np = np.asarray(labels, dtype=np.int64)
    cls_count = np.bincount(labels_np, minlength=2) + 1e-6
    cls_w = 1.0 / cls_count
    weights = cls_w[labels_np]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True), cls_count

@torch.no_grad()
def evaluate_auc(model, dct_branch, fuse_gate, head, dl_val, device="cuda"):
    model.eval(); dct_branch.eval(); fuse_gate.eval(); head.eval()
    all_y, s_rgb, s_freq, s_fuse = [], [], [], []

    for xv, yv in dl_val:
        xv = xv.to(device, non_blocking=True)            # (B,3,H,W) in [0,1]
        yv = yv.to(device, non_blocking=True).float()    # (B,)

        # 视觉向量 (B,D) float32
        f_rgb  = model.encode_image_feats_from_tensor(xv)      # -> (B,D)
        if f_rgb.dim() == 3:
            f_rgb = f_rgb.mean(dim=1)
        f_rgb = f_rgb.float()

        # 频域向量 (B,D) float32
        f_freq = dct_branch(xv).float()                        # -> (B,D)

        # 融合向量 (B,D) float32
        fused  = fuse_gate(f_rgb, f_freq).float()              # -> (B,D)

        # 同一线性头分别打分三路（便于横向对比）
        logit_rgb  = head(f_rgb).squeeze(-1)                   # (B,)
        logit_freq = head(f_freq).squeeze(-1)                  # (B,)
        logit_fuse = head(fused).squeeze(-1)                   # (B,)

        s_rgb.append(torch.sigmoid(logit_rgb).cpu())
        s_freq.append(torch.sigmoid(logit_freq).cpu())
        s_fuse.append(torch.sigmoid(logit_fuse).cpu())
        all_y.append(yv.cpu())

    y  = torch.cat(all_y).numpy()
    pr = torch.cat(s_rgb ).numpy()
    pf = torch.cat(s_freq).numpy()
    pu = torch.cat(s_fuse).numpy()

    def _metrics(s):
        return (roc_auc_score(y, s), average_precision_score(y, s), float(((s > 0.5) == y).mean()))

    m_rgb  = _metrics(pr)
    m_freq = _metrics(pf)
    m_fuse = _metrics(pu)

    # 恢复训练态（仅对可训练模块）
    dct_branch.train(); fuse_gate.train(); head.train()
    return m_rgb, m_freq, m_fuse

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-root", required=True, nargs="+", type=str,
                help="一个或多个真实帧根目录")
    ap.add_argument("--fake-root", required=True, nargs="+", type=str,
                help="一个或多个伪造帧根目录")

    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-train", type=int, default=128)
    ap.add_argument("--batch-val", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--k-keep", type=int, default=16, help="保留的 DCT 子带个数")
    ap.add_argument("--hidden", type=int, default=64, help="频域分支中的中间通道数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-int", type=int, default=20)
    ap.add_argument("--val-int", type=int, default=200)
    args = ap.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # 数据
    ds_full = FrameDataset(args.real_root, args.fake_root, args.img_size)
    N = len(ds_full)
    idx = np.arange(N)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    val_n = max(1, int(N * args.val_ratio))
    val_idx, train_idx = idx[:val_n], idx[val_n:]
    ds_train = Subset(ds_full, train_idx)
    ds_val   = Subset(ds_full, val_idx)

    train_labels = [ds_full.items[i][1] for i in train_idx]
    sampler, cls_count = build_balanced_sampler(train_labels)

    nw = 4
    dl_train = DataLoader(ds_train, batch_size=args.batch_train, sampler=sampler,
                          num_workers=nw, pin_memory=True, persistent_workers=(nw > 0))
    dl_val   = DataLoader(ds_val, batch_size=args.batch_val, shuffle=False,
                          num_workers=nw, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型主干（仅用视觉，不启用 LLM/LoRA）
    model = OpenLLAMAPEFTModel(
        imagesize=args.img_size,
        max_tgt_len=16,
        lora_r=0,
        enable_llm=False
    ).to(device).eval()

    # 动态获取视觉特征维度 D（通常为 1280）
    with torch.no_grad():
        _probe = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
        f_probe = model.encode_image_feats_from_tensor(_probe)
        if f_probe.dim() == 3:
            f_probe = f_probe.mean(dim=1)
        D = int(f_probe.shape[-1])

    print(f"[Info] visual_hidden_size = {D}, train frames={len(train_idx)}, val frames={len(val_idx)}")

    # 频域分支 + 门控融合（全部对齐到 D）
    dct_branch = DCTBranch(k_keep=args.k_keep, hidden=args.hidden, out_dim=D).to(device)
    fuse_gate  = GatedFuse(dim=D).to(device)

    # 线性探针头（共享给 rgb/freq/fuse 三路，便于横向对比）
    head = nn.Linear(D, 1).to(device)
    nn.init.xavier_normal_(head.weight, gain=2.0)
    nn.init.zeros_(head.bias)

    # 冻结主干
    for p in model.parameters():
        p.requires_grad_(False)

    # 只训练频域分支 + 门控 + 线性头
    dct_branch.train(); fuse_gate.train(); head.train()

    # 优化器 + CosineLR(warmup 10%)
    opt = optim.AdamW(
        list(dct_branch.parameters()) + list(fuse_gate.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=args.wd
    )
    total_steps = max(1, len(dl_train) * args.epochs)
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * t))

    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    bce = nn.BCEWithLogitsLoss()

    # 训练
    step = 0
    for ep in range(1, args.epochs + 1):
        for it, (x, y) in enumerate(dl_train, 1):
            step += 1
            x = x.to(device, non_blocking=True).float()              # (B,3,H,W)
            y = y.to(device, non_blocking=True).float().view(-1)     # (B,)

            # 视觉向量 (冻结，不反传)
            with torch.no_grad():
                f_rgb = model.encode_image_feats_from_tensor(x)      # (B,D) or (B,T,D)
                if f_rgb.dim() == 3:
                    f_rgb = f_rgb.mean(dim=1)
                f_rgb = f_rgb.float()

            # 频域向量 + 融合（参与训练）
            f_freq = dct_branch(x).float()                           # (B,D)
            fused  = fuse_gate(f_rgb, f_freq).float()                # (B,D)
            if fused.dim() > 2:
                fused = fused.reshape(fused.size(0), -1)

            logit = head(fused).squeeze(-1)                          # (B,)
            loss  = bce(logit, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),       1.0)
            torch.nn.utils.clip_grad_norm_(fuse_gate.parameters(),  1.0)
            torch.nn.utils.clip_grad_norm_(dct_branch.parameters(), 1.0)
            opt.step()
            scheduler.step()

            if step % args.log_int == 0:
                with torch.no_grad():
                    prob = torch.sigmoid(logit)
                    acc  = ((prob > 0.5).float() == y).float().mean().item()
                print(f"[ep {ep} | it {it} | step {step}] "
                      f"loss={loss.item():.4f} acc={acc:.3f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} B={y.numel()}")

            if step % args.val_int == 0:
                m_rgb, m_freq, m_fuse = evaluate_auc(model, dct_branch, fuse_gate, head, dl_val, device)
                print(f"[VAL @ step {step}] "
                      f"rgb  AUC={m_rgb[0]:.4f} AP={m_rgb[1]:.4f} ACC@0.5={m_rgb[2]:.3f}  | "
                      f"freq AUC={m_freq[0]:.4f} AP={m_freq[1]:.4f} ACC@0.5={m_freq[2]:.3f}  | "
                      f"fuse AUC={m_fuse[0]:.4f} AP={m_fuse[1]:.4f} ACC@0.5={m_fuse[2]:.3f}")

    # 最终验证
    m_rgb, m_freq, m_fuse = evaluate_auc(model, dct_branch, fuse_gate, head, dl_val, device)
    print(f"[FINAL VAL] "
          f"rgb  AUC={m_rgb[0]:.4f} AP={m_rgb[1]:.4f} ACC@0.5={m_rgb[2]:.3f}  | "
          f"freq AUC={m_freq[0]:.4f} AP={m_freq[1]:.4f} ACC@0.5={m_freq[2]:.3f}  | "
          f"fuse AUC={m_fuse[0]:.4f} AP={m_fuse[1]:.4f} ACC@0.5={m_fuse[2]:.3f}")

    # 保存探针（head）与门控/频域权重（state_dict 形式，后续加载要先实例化再 load_state_dict）
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True, parents=True)
    torch.save(head.state_dict(), ckpt_dir / "freq_head.pt")
    torch.save({
        "dct_branch": dct_branch.state_dict(),
        "fuse_gate":  fuse_gate.state_dict(),
        "dim": D,
        "k_keep": args.k_keep,
        "hidden": args.hidden,
    }, ckpt_dir / "freq_modules.pt")
    print(f"[SAVE] checkpoints/freq_head.pt & checkpoints/freq_modules.pt saved.")

if __name__ == "__main__":
    main()
