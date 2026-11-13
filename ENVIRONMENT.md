# 训练环境记录

> 记录日期：2026-11-13
> 用途：BreakHis / BACH 实验训练与评估

## 硬件

| 项目 | 配置 |
|---|---|
| CPU | 2 × Intel Xeon Gold 6530，128 vCPU |
| 内存 | 1.0 TiB（可用约 920 GiB） |
| GPU | NVIDIA RTX PRO 6000 Blackwell，96 GB VRAM |
| 驱动 / CUDA | Driver 595.84 / CUDA 13.2 |
| 存储 | 本地 894 GB（已用 79%）+ 2 × 7 TB NVMe + 284 TB 共享文件系统 |
| 系统负载 | 10.30 / 6.62 / 5.66（128 vCPU 下负载很低，可正常训练） |
| 运行时长 | 114 天+ |

## 对实验的影响

- **96 GB 显存**：BreakHis/BACH 训练、较大 batch（64–256）、Swin/ViT/Mamba 类模型都能放下；可以做全分辨率或较高分辨率实验。
- **128 vCPU + 1 TiB 内存**：DataLoader 可设 8–16 workers；预 resize 缓存可完全放内存或页缓存。
- **本地盘**：894 GB 已用 79%，剩余约 188 GB；数据集约 24 GB、缓存约 1–2 GB、checkpoint 需要控制数量并定期清理。
- **系统负载**：1 分钟负载约 10（128 vCPU），仍有充足余量。

## 存储建议

| 用途 | 建议位置 |
|---|---|
| 项目代码 | 共享文件系统或 NVMe（按团队约定） |
| 原始数据集 | NVMe（约 24 GB） |
| resize 缓存 | NVMe（约 1–2 GB） |
| checkpoint / 训练日志 | NVMe，定期同步到共享文件系统 |
| 最终结果 / 论文材料 | 共享文件系统 |

训练脚本建议通过配置指定 `data_root`、`cache_root`、`run_root`，不要把服务器绝对路径写死。

## 软件与兼容性检查

Blackwell（RTX PRO 6000）需要较新的 PyTorch / CUDA 组合。首次上机后先执行：

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

确认：

- `torch.cuda.is_available()` 为 `True`
- `torch.version.cuda` 与驱动兼容
- 能正常分配显存并跑一个小矩阵乘法

如果 PyTorch 不识别该 GPU，优先升级到支持 Blackwell（sm_120）的最新稳定版或官方 nightly，而不是降级驱动。

## 训练配置建议

- 混合精度：`torch.amp.autocast("cuda")` + `GradScaler`
- 固定随机种子；主结果报告 5 折 mean ± std
- 输入尺寸固定时可开 `torch.backends.cudnn.benchmark = True`
- DataLoader：`num_workers=8–16`、`pin_memory=True`、`persistent_workers=True`
- 先用小规模 smoke test 验证流程，再放大 batch 和 epoch

## 数据同步

本机（macOS，Apple M4 / MPS）已下载并校验数据与划分。训练服务器需要：

1. 同步 `Dataset/BreakHis/BreaKHis_v1`、`Dataset/BACH/ICIAR2018_BACH_Challenge`；
2. 运行 `python3 cli/build_dataset_splits.py` 重新生成划分（或直接同步 `Dataset/*/splits/`）；
3. 运行 `python3 cli/validate_datasets.py` 确认无缺失。

注意：`Dataset/` 已在 `.gitignore` 中，不走 Git 传输，建议用 rsync / scp 或服务器重新下载。

## 备注

- 本记录只包含硬件与软件环境，不包含主机名、IP、账号或密钥。
- 实际挂载点、项目路径和调度系统（如 Slurm）在首次上机后补充。
