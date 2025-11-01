# infer_freq_only.py
import os, sys, csv, argparse
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from sklearn.metrics import roc_auc_score, average_precision_score

from model.freq_branch import DCTBranch          # 频域分支
from utils.freq import block_dct_maps            # 8x8 DCT -> K 子带

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# -------- IO --------
def list_images(root: str) -> List[str]:
    p = Path(root)
    files = [str(x) for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS]
    files.sort()
    return files

def load_batch(paths: List[str], img_size: int) -> torch.Tensor:
    ims = []
    for fp in paths:
        im = Image.open(fp).convert("RGB").resize((img_size, img_size))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)
        ims.append(t)
    return torch.stack(ims, dim=0)  # (B,3,H,W)

def list_with_labels(real_root: str, fake_root: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str,int]] = []
    for root, lab in [(real_root,0), (fake_root,1)]:
        root_p = Path(root)
        if not root_p.exists():
            raise FileNotFoundError(f"Not found: {root}")
        for fp in root_p.rglob("*"):
            if fp.suffix.lower() in IMG_EXTS:
                items.append((str(fp), lab))
    items.sort(key=lambda x: x[0])
    return items

# -------- Heatmap --------
@torch.no_grad()
def make_heatmap_from_dct(x: torch.Tensor, k_keep: int, out_size: int) -> List[np.ndarray]:
    dct_map = block_dct_maps(x, k_keep=k_keep)             # (B,K,H/8,W/8) float32
    heat = dct_map.abs().mean(dim=1, keepdim=True)         # (B,1,h,w)
    heat = F.interpolate(heat, size=(out_size, out_size), mode="bilinear", align_corners=False)  # (B,1,S,S)
    heat = heat.squeeze(1).cpu().numpy()                   # (B,S,S)
    outs = []
    for m in heat:
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        m = (m * 255).astype(np.uint8)
        outs.append(m)
    return outs

# -------- Load ckpt --------
def load_checkpoints(mod_path: str, head_path: str, k_keep_arg: int, hidden_arg: int, device: str):
    D = 1280
    k_keep = k_keep_arg
    hidden = hidden_arg

    dct_branch = DCTBranch(k_keep=k_keep, hidden=hidden, out_dim=D).to(device)
    head = torch.nn.Linear(D, 1).to(device)

    if Path(mod_path).exists():
        mod_ckpt = torch.load(mod_path, map_location=device)
        if "dim"    in mod_ckpt: D = int(mod_ckpt["dim"])
        if "k_keep" in mod_ckpt: k_keep = int(mod_ckpt["k_keep"])
        if "hidden" in mod_ckpt: hidden = int(mod_ckpt["hidden"])
        dct_branch = DCTBranch(k_keep=k_keep, hidden=hidden, out_dim=D).to(device)
        dct_branch.load_state_dict(mod_ckpt["dct_branch"])
        if "fuse_gate" in mod_ckpt:
            print("[info] fuse_gate 存在但本脚本不会用到（仅频域推理）。")
        print(f"[load] freq_modules from {mod_path} | dim={D} k_keep={k_keep} hidden={hidden}")
    else:
        print(f"[warn] {mod_path} 不存在，使用随机初始化的 DCTBranch (dim={D}, k_keep={k_keep}, hidden={hidden})")

    if Path(head_path).exists():
        sd = torch.load(head_path, map_location=device)
        head.load_state_dict(sd)  # 你训练时保存的是 state_dict
        print(f"[load] head from {head_path}")
    else:
        print(f"[warn] {head_path} 不存在，使用随机初始化的线性头")

    dct_branch.eval()
    head.eval()
    return dct_branch, head, D, k_keep, hidden

# -------- Metrics --------
def cls_metrics(y: np.ndarray, s: np.ndarray) -> Dict[str, float]:
    auc = roc_auc_score(y, s)
    ap  = average_precision_score(y, s)
    acc = float(((s > 0.5) == y).mean())
    return {"auc": float(auc), "ap": float(ap), "acc": acc}

# -------- Main --------
def main():
    ap = argparse.ArgumentParser()
    # 二选一：评估(有标签) 或 单目录推理(无标签)
    ap.add_argument("--input", type=str, default=None, help="单目录推理（无标签，不计算AUC）")
    ap.add_argument("--real-root", type=str, default=None, help="真实帧根目录（有标签，计算AUC）")
    ap.add_argument("--fake-root", type=str, default=None, help="伪造帧根目录（有标签，计算AUC）")

    ap.add_argument("--output", type=str, default="./output_freq", help="输出根目录")
    ap.add_argument("--ckpt-mod", type=str, default="checkpoints/freq_modules.pt", help="包含 dct_branch 的权重")
    ap.add_argument("--ckpt-head", type=str, default="checkpoints/freq_head.pt", help="线性头权重")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--k-keep", type=int, default=16, help="若无 ckpt，用该值初始化 DCTBranch")
    ap.add_argument("--hidden", type=int, default=64, help="若无 ckpt，用该值初始化 DCTBranch")
    ap.add_argument("--thr", type=float, default=0.5, help="prob>=thr 视为 deepfake")
    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "max"], help="video 聚合方式")
    ap.add_argument("--out-frame", type=str, default=None, help="frame 级 CSV 路径")
    ap.add_argument("--out-video", type=str, default=None, help="video 级 CSV 路径（需有标签模式）")
    ap.add_argument("--no-heatmap", action="store_true", help="不保存热图以加速")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "anomaly").mkdir(parents=True, exist_ok=True)

    dct_branch, head, D, k_keep, hidden = load_checkpoints(
        mod_path=args.ckpt_mod,
        head_path=args.ckpt_head,
        k_keep_arg=args.k_keep,
        hidden_arg=args.hidden,
        device=device,
    )

    # ===== 有标签评估（推荐） =====
    if args.real_root and args.fake_root:
        items = list_with_labels(args.real_root, args.fake_root)   # [(path,label), ...]
        print(f"[info] found {len(items)} frames | real={sum(1 for _,l in items if l==0)} fake={sum(1 for _,l in items if l==1)}")

        all_probs, all_labels, all_videos, all_paths = [], [], [], [p for p,_ in items]

        for i in range(0, len(items), args.batch):
            chunk_items = items[i:i+args.batch]
            chunk_paths = [p for p,_ in chunk_items]
            x = load_batch(chunk_paths, args.img_size).to(device).float()

            with torch.no_grad():
                f_freq = dct_branch(x)                        # (B,D) float32
                logit  = head(f_freq).squeeze(-1)             # (B,)
                prob   = torch.sigmoid(logit).cpu().numpy()   # (B,)
                heatmaps = None
                if not args.no_heatmap:
                    heatmaps = make_heatmap_from_dct(x, k_keep=k_keep, out_size=args.img_size)

            for j, (fp, lab) in enumerate(chunk_items):
                all_probs.append(float(prob[j]))
                all_labels.append(int(lab))
                all_videos.append(Path(fp).parent.name)
                if heatmaps is not None:
                    save_name = f"{i + j + 1:06d}.png"
                    cv2.imwrite(str(out_root / "anomaly" / save_name), heatmaps[j])

            print(f"[{i+len(chunk_items)}/{len(items)}] done")

        y = np.asarray(all_labels, dtype=np.int64)
        s = np.asarray(all_probs,  dtype=np.float32)

        # frame CSV
        out_frame = args.out_frame or (out_root / "freq_frame.csv")
        with open(out_frame, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["path","video","score","label"])
            for p, v, sc, lb in zip(all_paths, all_videos, s, y):
                w.writerow([p, v, f"{sc:.6f}", lb])
        print(f"[save] frame csv -> {out_frame}")

        # frame 指标
        m_frame = cls_metrics(y, s)
        print(f"[FRAME] AUC={m_frame['auc']:.4f} AP={m_frame['ap']:.4f} ACC@0.5={m_frame['acc']:.3f}")

        # video 聚合
        vid_scores: Dict[str, List[float]] = {}
        vid_labels: Dict[str, List[int]]   = {}
        for v, sc, lb in zip(all_videos, s, y):
            vid_scores.setdefault(v, []).append(sc)
            vid_labels.setdefault(v, []).append(lb)

        vids, v_s, v_y = [], [], []
        for v in sorted(vid_scores.keys()):
            scores = np.asarray(vid_scores[v], dtype=np.float32)
            labs   = np.asarray(vid_labels[v], dtype=np.int64)
            agg = float(scores.mean() if args.agg == "mean" else scores.max())
            lab = int(np.mean(labs) >= 0.5)  # 多数为 1 则 1
            vids.append(v); v_s.append(agg); v_y.append(lab)

        v_s = np.asarray(v_s, dtype=np.float32)
        v_y = np.asarray(v_y, dtype=np.int64)
        m_video = cls_metrics(v_y, v_s)
        print(f"[VIDEO] AUC={m_video['auc']:.4f} AP={m_video['ap']:.4f} ACC@0.5={m_video['acc']:.3f} (agg={args.agg})")

        # video CSV
        out_video = args.out_video or (out_root / "freq_video.csv")
        with open(out_video, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["video","score","label","n_frames"])
            for v in vids:
                scores = vid_scores[v]
                sc = np.mean(scores) if args.agg=='mean' else np.max(scores)
                w.writerow([v, f"{sc:.6f}", int(np.mean(vid_labels[v])>=0.5), len(scores)])
        print(f"[save] video csv -> {out_video}")
        return

    # ===== 无标签推理 =====
    if args.input is None:
        print("[error] 请提供 --real-root/--fake-root 进行评估，或提供 --input 做无标签推理。")
        sys.exit(1)

    paths = list_images(args.input)
    if not paths:
        print(f"[error] no images found under: {args.input}")
        sys.exit(1)

    csv_path = out_root / "results_freq_only.csv"
    rows = []
    for i in range(0, len(paths), args.batch):
        chunk = paths[i:i + args.batch]
        x = load_batch(chunk, args.img_size).to(device).float()

        with torch.no_grad():
            f_freq = dct_branch(x)                        # (B,D)
            logit  = head(f_freq).squeeze(-1)             # (B,)
            prob   = torch.sigmoid(logit).cpu().numpy()   # (B,)
            heatmaps = None
            if not args.no_heatmap:
                heatmaps = make_heatmap_from_dct(x, k_keep=k_keep, out_size=args.img_size)

        for j, fp in enumerate(chunk):
            pred = int(prob[j] >= args.thr)
            if heatmaps is not None:
                save_name = f"{i + j + 1:06d}.png"
                cv2.imwrite(str(out_root / "anomaly" / save_name), heatmaps[j])
            print(f"[{i+j+1}/{len(paths)}] {fp} | prob={prob[j]:.4f} | pred={pred}")
            rows.append([i + j + 1, fp, f"{prob[j]:.6f}", pred])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["index", "file", "fakeprob", "pred"]); w.writerows(rows)
    print(f"[DONE] CSV -> {csv_path}")
    print(f"[DONE] heatmaps -> {out_root/'anomaly'}")

if __name__ == "__main__":
    main()
