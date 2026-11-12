# CLAUDE.md

加载Master_Plan中的最新版本计划

## 目录结构

```text
BreakHis_BACH/
├── CLAUDE.md                 # 本文件：Claude Code 项目上下文
├── README.md                 # 项目根目录说明
├── .claude/
│   └── settings.json         # 项目级 Claude Code 配置
├── .gitignore                # 忽略 .idea/、.DS_Store、._* 等
├── Master_Plan/
│   ├── README.md             # 计划版本管理说明
│   └── plan_v1.md            # 论文研究计划书 v1（当前最新版）
├── Dataset/                  # 数据集保存目录
│   ├── README.md
│   ├── BreakHis/             # BreakHis 主数据集
│   └── BACH/                 # BACH 外部验证集
├── Resources/                # 所有外部参考资源保存目录
│   ├── README.md
│   ├── Patho-LiteDFSNet/     # 轻量病理架构参考文献
│   ├── MSWA-ResNet/          # 患者级分类参考文献
│   └── Others/               # 其他参考论文
├── src/                      # 所有源代码保存目录
│   └── README.md
├── cli/                      # 所有代码执行入口
│   └── README.md
├── test/                     # 测试代码（单元测试、集成测试等）
│   └── README.md
├── Draft/                    # 论文撰写保存目录
│   └── README.md
├── .idea/                    # JetBrains IDE 配置，非项目内容
└── ._*                       # macOS 元数据（AppleDouble），非项目内容
```

## 工作约定

1. **计划版本**：`Master_Plan/` 采用 `plan_v[版本号].md` 迭代，最新版本为准。更新计划时新建 `plan_vN.md`，保留旧版本；必要时同步更新 `Master_Plan/README.md`。
2. **代码位置**：源代码放 `src/`，可执行入口与脚本放 `cli/`，测试放 `test/`，不在根目录散放代码。
3. **数据位置**：数据集放 `Dataset/`，BreakHis 与 BACH 分目录保存；数据文件一般不提交到 Git，如需忽略应在 `.gitignore` 中补充。
4. **参考资料**：论文 PDF 等外部资源放 `Resources/`，按方法或主题建子目录（现有 `Patho-LiteDFSNet/`、`MSWA-ResNet/`、`Others/`）。
5. **论文撰写**：草稿、图表与成稿放 `Draft/`。
6. **文档语言**：中文正文，术语与英文缩写按计划书原文保留。
7. **实验严谨性底线**（见 `Master_Plan/plan_v1.md`）：
   - 使用 BreakHis-IB 患者级划分 CSV，不使用图像级划分；五折患者级交叉验证。
   - 主任务为患者级二分类（良性 vs 恶性），八分类仅作补充或消融。
   - 指标按 40× / 100× / 200× / 400× 分层报告，不混合平均。
   - BACH 只作外部验证，不参与调参；采用两阶段（零样本 + 冻结头校准）。
   - 统计检验用五折配对 Wilcoxon 或 Bootstrap 95% CI，不伪造显著性。
8. **不改动既有文档**：现有说明与计划文件保持原样，除非用户明确要求修改。
9. **Git**：远程仓库为 `git@github.com:XiaoHG/BreakHis_BACH.git`，主分支 `main`。
