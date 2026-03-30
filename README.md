# FREQ-DFD

基于**频域信息（DCT 子带）**的 Deepfake 检测基线工程，提供训练、评估、无标签推理与热图导出能力。

## 1. 项目简介

FREQ-DFD 关注这样一个问题：仅看 RGB 空间时，一些伪造痕迹会被纹理、光照和压缩噪声掩盖；而在频域中，伪造模型引入的重采样、插值和高频分布异常往往更容易暴露。

核心直觉：
- 将图像切分为块并映射到 DCT 频域后，可观察到伪造样本在若干频带上的能量分布偏移。
- 使用轻量频域分支学习这些差异，再通过线性头完成真假判别。
- 在评估/推理时可把频域响应重投影成热图，直观显示“异常区域”。

## 2. 核心特性

- **仅频域推理（Freq-Only Inference）**：`infer_freq_only.py` 可在无 RGB 融合分支参与下独立完成推理。
- **融合训练（RGB + Freq）**：`train_freq_head.py` 训练时使用视觉特征与频域特征门控融合，增强稳健性。
- **可导出热图（Heatmap）**：推理默认生成 `anomaly/*.png`，用于定位频域异常区域。

## 3. 环境安装

推荐环境：
- Python **3.10+**
- PyTorch **2.1+**（建议与 CUDA 版本匹配）

依赖安装（复制即用）：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pillow opencv-python scikit-learn
```

> 若仅 CPU 运行，请将 PyTorch 安装命令替换为 CPU 轮子（官方指引见 PyTorch 官网），其余命令不变。

GPU/CPU 说明：
- 脚本自动选择设备：有 CUDA 则使用 GPU，否则回退 CPU。
- CPU 可运行但速度显著慢于 GPU，尤其在大批量推理/训练时。

## 4. 数据准备

训练与有标签评估都要求 `real` / `fake` 两类目录。推荐目录结构：

```text
data/
  real/
    video_001/
      000001.png
      000002.png
    video_002/
      ...
  fake/
    video_101/
      000001.png
      000002.png
    video_102/
      ...
```

约定：
- `real` 下为真实帧；`fake` 下为伪造帧。
- 每个子目录通常对应一个视频 ID（用于 video-level 聚合统计）。

## 5. 快速开始

可直接使用封装脚本（推荐）：

```bash
bash scripts/quickstart.sh
```

也可按最小命令分别运行：

- 训练

```bash
python train_freq_head.py --real-root /path/to/data/real --fake-root /path/to/data/fake
```

- 有标签评估

```bash
python infer_freq_only.py --real-root /path/to/data/real --fake-root /path/to/data/fake
```

- 无标签推理

```bash
python infer_freq_only.py --input /path/to/images_or_frames
```

## 6. 输出说明

默认输出目录为 `./output_freq`：

- `freq_frame.csv`（有标签模式）
  - `path`：帧文件路径
  - `video`：帧所属视频目录名
  - `score`：伪造概率（0~1）
  - `label`：真实标签（`0=real, 1=fake`）

- `freq_video.csv`（有标签模式）
  - `video`：视频 ID
  - `score`：视频聚合分数（`--agg mean/max`）
  - `label`：视频标签（多数帧规则）
  - `n_frames`：该视频参与统计的帧数

- `anomaly/*.png`
  - 频域异常热图，数值归一化到 0~255 后保存为灰度图。

无标签模式会输出 `results_freq_only.csv`：
- `index`：序号
- `file`：图像路径
- `fakeprob`：伪造概率
- `pred`：二分类结果（默认阈值 `0.5`）

## 7. 可视化示例

推理时可配合以下三联图进行展示：
1. 输入图（RGB）
2. 异常热图（`anomaly/*.png`）
3. 预测分数（如 `fakeprob=0.873`）

你可以在报告中采用如下描述模板：

```text
sample: video_101/000123.png | fakeprob=0.873 | pred=1
```

## 8. 常见报错与排查

- CUDA 不可用
  - 现象：脚本回退到 CPU，速度较慢。
  - 排查：`python -c "import torch; print(torch.cuda.is_available())"`
  - 处理：检查驱动/CUDA/PyTorch 版本匹配，或先用 CPU 小规模验证流程。

- 模型权重缺失
  - 现象：提示 `checkpoints/freq_modules.pt` 或 `checkpoints/freq_head.pt` 不存在。
  - 影响：脚本会使用随机初始化权重，结果仅用于联调，不具参考价值。
  - 处理：先完成训练，或通过 `--ckpt-mod` / `--ckpt-head` 指向正确权重。

- 路径错误 / 数据为空
  - 现象：`Not found`、`no images found`、样本数量为 0。
  - 排查：确认 `--real-root`、`--fake-root`、`--input` 指向存在目录，且包含图片后缀（png/jpg/jpeg/bmp/webp）。

## 9. 致谢与引用

感谢开源社区提供的 PyTorch、OpenCV 等基础组件。若本项目对你的研究或工程有帮助，可引用：

```bibtex
@misc{freqdfd2026,
  title        = {FREQ-DFD: Frequency-Domain Baseline for Deepfake Detection},
  author       = {FREQ-DFD Contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/your-org/FREQ-DFD}}
}
```

---

如需一键运行最常见流程，请直接查看并修改 `scripts/quickstart.sh` 中的路径变量。
