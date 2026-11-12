# 说明
- 保存数据集目录；

## 数据状态与划分

### BreakHis
- `BreakHis/BreaKHis_v1/`：7,909 张 8-bit RGB PNG（2,480 良性 / 5,429 恶性），81 个唯一 slide ID。
- 74 张 700×456 图片均来自 slide 14-12465（乳头状癌），其余为 700×460。
- `BreakHis/Folds.csv`：Kaggle 附带划分，患者 14-13412 在 fold 1/3/5 跨 train/test，存在患者级泄漏，**不要用于正式实验**。
- `BreakHis/splits/patient_5fold.csv`：正式患者级五折划分（seed=42，已校验无泄漏）。
- 稀有亚型无法保证每折齐全（A 仅 4 例、PT 3 例、LC 5 例）：当前划分为 A 缺 1 折、PT 缺 2 折、LC 缺 1 折。亚型指标建议使用汇总的 out-of-fold 预测，而不是逐折宏平均。
- 本项目统一使用患者级五折划分；由 `cli/build_dataset_splits.py` 生成，固定 seed=42，已校验无患者泄漏。
- 来源为 Kaggle 镜像 `ambarish/breakhis`；数据仅限非商业研究，引用 Spanhol et al., IEEE TBME 2016。

### BACH
- `BACH/ICIAR2018_BACH_Challenge/.../Photos/`：400 张 2048×1536 TIFF，四类各 100 张。
- 位深混合：381 张 16-bit，19 张 8-bit（b091-094、is081-084、iv077-087，均在 train）；读入时需逐图按位深归一化。
- `BACH/splits/eva_patient_split.csv`：采用 EVA 患者级索引，268 train / 132 val。
- 官方患者表 `ICIAR2018_BACH_dataset_micro_patient.xlsx` 仅填写 145/400，故不自行推断患者号。
- 许可：CC BY-NC-ND 4.0；引用 Aresta et al. 2019。官方包见 Zenodo record 3632035。

### 生成与校验
```bash
python3 cli/build_dataset_splits.py
python3 cli/validate_datasets.py
python3 -m unittest discover -s test -v
```