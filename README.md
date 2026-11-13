# BreakHis_BACH

面向小样本场景的轻量化可解释乳腺癌组织病理图像分类研究。主数据集为 **BreakHis**，外部验证集为 **BACH**；核心方法是双路轻量编码器 + 不确定性感知门控融合；评估以**患者级五折交叉验证**为严谨性底线。

当前阶段：数据下载与校验、患者级划分、训练/评估流水线已完成；fold 0 pilot 基线已产出。下一步在服务器上跑全量 5 折基线。

## 相关文档

- 研究计划：[Master_Plan/plan_v1.md](Master_Plan/plan_v1.md)
- Claude 项目上下文：[CLAUDE.md](CLAUDE.md)
- 训练服务器环境：[ENVIRONMENT.md](ENVIRONMENT.md)
- 数据集说明：[Dataset/README.md](Dataset/README.md)

## 项目结构

```text
BreakHis_BACH/
├── README.md
├── CLAUDE.md                 # Claude Code 项目上下文
├── ENVIRONMENT.md            # 训练服务器环境记录
├── requirements.txt          # 训练依赖
├── Master_Plan/              # 研究计划（plan_v1.md 为当前版本）
├── Dataset/                  # 数据集与划分（不入 Git）
│   ├── BreakHis/
│   │   ├── BreaKHis_v1/
│   │   └── splits/
│   └── BACH/
│       ├── ICIAR2018_BACH_Challenge/
│       ├── ICIAR2018_BACH_Challenge_TestDataset/
│       └── splits/
├── Resources/                # 参考文献 PDF
├── Draft/                    # 论文撰写
├── src/
│   ├── data/                 # 数据解析、划分、PyTorch Dataset
│   ├── models/               # 基线模型
│   └── engine/               # 训练循环、指标
├── cli/
│   ├── build_dataset_splits.py
│   ├── validate_datasets.py
│   └── train.py
└── test/                     # 单元测试
```

## 数据状态

### BreakHis（主数据集）

- 7,909 张 8-bit RGB PNG：良性 2,480 / 恶性 5,429
- 4 个放大倍数：40× 1,995、100× 2,081、200× 2,013、400× 1,820
- 8 个组织学亚型；82 个 slide 目录，按 slide ID 去重后 **81 个唯一病例**
- 74 张图片为 700×456（均来自 slide 14-12465），其余 700×460
- 划分文件：`Dataset/BreakHis/splits/patient_5fold.csv`
- **不要使用** `Dataset/BreakHis/Folds.csv`：患者 14-13412 在部分折跨 train/test，存在泄漏

### BACH（外部验证集）

- 400 张标注 H&E TIFF，四类各 100：Benign / InSitu / Invasive / Normal
- 尺寸 2048×1536；位深混合：381 张 16-bit、19 张 8-bit（均在 train）
- 另有 100 张无标签官方测试图，不参与评估
- 划分文件：`Dataset/BACH/splits/eva_patient_split.csv`（EVA 患者级索引，268 train / 132 val）
- 任务映射：Normal + Benign → 良性；InSitu + Invasive → 恶性
- 许可：CC BY-NC-ND 4.0；引用 Aresta et al. 2019

`Dataset/` 已在 `.gitignore` 中，数据不走 Git 传输；请用 rsync/scp 或重新下载。

## 环境

### 本机

- macOS，Apple M4，PyTorch 2.11 + MPS
- 用途：数据处理、原型开发、冒烟测试、单折 pilot
- 不建议在本机并行跑全量 5 折

### 训练服务器

| 项目 | 配置 |
|---|---|
| CPU | 2 × Intel Xeon Gold 6530，128 vCPU |
| 内存 | 1.0 TiB |
| GPU | NVIDIA RTX PRO 6000 Blackwell，96 GB VRAM |
| 驱动 / CUDA | Driver 595.84 / CUDA 13.2 |
| 存储 | 本地 894 GB + 2 × 7 TB NVMe + 284 TB 共享文件系统 |

详细说明见 [ENVIRONMENT.md](ENVIRONMENT.md)。

## 安装

### 1. Python 环境

建议 Python 3.10+：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2. PyTorch 与依赖

Blackwell（RTX PRO 6000）需要较新的 PyTorch/CUDA 组合。先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装与 Driver 595.84 / CUDA 13.2 兼容的版本，例如：

```bash
# 示例：请根据实际可用的 CUDA wheel 调整 cu128 / cu130
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

再安装其余依赖：

```bash
pip install numpy pillow scikit-learn tqdm pyyaml
# 或使用 requirements.txt（注意不要覆盖刚装好的 CUDA 版 torch）
# pip install -r requirements.txt
```

### 3. 验证

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

确认 `torch.cuda.is_available()` 为 `True`，且能正常分配显存。

## 数据准备

先从本机同步数据（或直接在服务器重新下载）：

```bash
rsync -av Dataset/ user@server:/path/to/BreakHis_BACH/Dataset/
```

在服务器上生成并校验划分：

```bash
python3 cli/build_dataset_splits.py
python3 cli/validate_datasets.py
```

预期：BreakHis 7,909 张、81 个唯一病例；BACH 400 张、268/132 划分；无患者泄漏。

## 训练

### 冒烟测试

```bash
python3 cli/train.py --model resnet50 --folds 0 --epochs 1 \
  --max-batches 10 --batch-size 16 --workers 0 --no-pretrained
```

### 单折 pilot

```bash
python3 cli/train.py --model resnet50 --folds 0 --epochs 5 \
  --batch-size 32 --workers 4 --out-dir runs/pilot

python3 cli/train.py --model efficientnet_b0 --folds 0 --epochs 5 \
  --batch-size 32 --workers 4 --out-dir runs/pilot
```

### 全量 5 折（服务器）

```bash
python3 cli/train.py --model resnet50 --epochs 20 --batch-size 64 \
  --workers 12 --val-fraction 0.2 --out-dir runs/baseline

python3 cli/train.py --model efficientnet_b0 --epochs 20 --batch-size 64 \
  --workers 12 --val-fraction 0.2 --out-dir runs/baseline
```

输出：

```text
runs/<model>/
├── summary.json          # 5 折 mean ± std + 分倍数结果
└── foldN/
    ├── best.pt
    ├── history.csv
    └── test_metrics.json
```

评估指标：Accuracy、Macro-F1、Balanced Accuracy、AUC、敏感性、特异性，并按 40×/100×/200×/400× 分层报告。

### 测试

```bash
python3 -m unittest discover -s test -v
```

## 当前 pilot 结果

fold 0，5 epochs，ImageNet 预训练；**仅用于验证流水线，不是最终结果**：

| 模型 | Accuracy | Macro-F1 | Balanced Acc | AUC | 敏感性 | 特异性 |
|---|---|---|---|---|---|---|
| ResNet-50 | 0.8795 | 0.8576 | 0.8580 | 0.9244 | 0.9127 | 0.8033 |
| EfficientNet-B0 | 0.8727 | 0.8492 | 0.8490 | 0.9196 | 0.9091 | 0.7889 |

按放大倍数（Accuracy）：

| 模型 | 40× | 100× | 200× | 400× |
|---|---|---|---|---|
| ResNet-50 | 0.8845 | 0.8735 | 0.9065 | 0.8496 |
| EfficientNet-B0 | 0.8722 | 0.8806 | 0.8993 | 0.8329 |

注意：验证集只有 6 位患者，选模偏乐观；test fold 有 16 位患者，以上为 test 指标。全量 5 折 + 更多 epochs 后再作为正式基线。

## 实验协议

- **患者级五折交叉验证**：以 slide ID 为患者标识，同一患者只出现在一个折中
- 测试折完全不参与训练与选模；验证集从训练折内按患者抽取
- 报告五折 mean ± std，并按放大倍数分层
- 不使用图像级划分
- 不依赖 BreakHis-IB 官方 CSV；划分由 `cli/build_dataset_splits.py` 固定 seed 生成
- BACH 使用 EVA 患者级索引（268/132）；读入时按位深归一化
- 统计检验：Bootstrap 95% CI / 效应量为主，Wilcoxon 作为辅助

## 注意事项

- 读图时排除 macOS `._*` 元数据文件
- BACH 381 张 16-bit + 19 张 8-bit，必须逐图归一化
- BreakHis 74 张 700×456，统一 resize
- `Dataset/`、`runs/`、`*.pt` 已在 `.gitignore` 中，不要提交数据和 checkpoint
- 服务器上首次运行先做 PyTorch/Blackwell 兼容性检查

## 引用

- Spanhol F A, et al. A Dataset for Breast Cancer Histopathological Image Classification. IEEE TBME, 2016.
- Aresta G, et al. BACH: Grand Challenge on Breast Cancer Histology Images. Medical Image Analysis, 2019.
