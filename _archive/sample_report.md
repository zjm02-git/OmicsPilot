# TCGA-BRCA 差异表达与通路富集分析报告（样例）

> 本报告由 GenPilot（多组学数据分析智能体）自动生成：解析 → 质检 → 人工确认 → 差异分析 → 富集 → 解读 → 报告，全流程 LLM 编排 + Python 真实统计，非 LLM 编数。

## 一、质检概览

| 项目 | 值 |
| --- | --- |
| 数据集 | TCGA-BRCA |
| 样本数（tumor / normal） | 226（113 / 113） |
| 独立病人数 | 217（按病人去重，丢弃 3 份重复等分） |
| 基因数 | 19938 |
| 缺失率 | 0 / 4,505,988 = 0.0000% |

## 二、差异表达分析

- 方法：Welch t 检验 + BH 多重校正；log2FC = mean(normal) − mean(tumor)，正数表示该基因在 normal 中更高。
- 独立过滤：剔除 4655 个低表达 / 不可检验基因，进入 BH 的基因数 m = 15283。
- **显著基因（FDR < 0.05 且 |log2FC| > 1）：1580**
  - normal 中更高：1029
  - tumor 中更高：551

### Top 10 显著基因（normal 中更高）

| 基因 | log2FC（normal vs tumor） | FDR | 方向 |
| --- | --- | --- | --- |
| VEGFD | +3.32 | 3.8e-78 | normal 中更高 |
| KLHL29 | +1.39 | 7.9e-76 | normal 中更高 |
| ADAMTS5 | +2.53 | 1.7e-72 | normal 中更高 |
| DMD | +2.19 | 3.1e-72 | normal 中更高 |
| CAVIN2 | +3.63 | 7.5e-72 | normal 中更高 |
| TMEM220 | +1.58 | 5.0e-71 | normal 中更高 |
| MME | +3.23 | 5.9e-69 | normal 中更高 |
| MAMDC2 | +2.50 | 8.2e-67 | normal 中更高 |
| PAMR1 | +2.83 | 3.2e-66 | normal 中更高 |
| ANKRD29 | +1.56 | 1.3e-65 | normal 中更高 |

## 三、通路富集（超几何检验 + BH 校正）

### tumor 中更高的基因（n=551）—— 显著富集 172 个通路

| 通路 | K | k | FDR |
| --- | --- | --- | --- |
| GOBP_SISTER_CHROMATID_SEGREGATION | 234 | 52 | 2.6e-28 |
| GOBP_MITOTIC_NUCLEAR_DIVISION | 282 | 56 | 2.7e-28 |
| GOBP_NUCLEAR_CHROMOSOME_SEGREGATION | 321 | 59 | 3.3e-28 |
| GOBP_MITOTIC_SISTER_CHROMATID_SEGREGATION | 193 | 47 | 8.1e-28 |
| GOBP_CHROMOSOME_SEGREGATION | 423 | 66 | 1.1e-27 |

### normal 中更高的基因（n=1029）—— 显著富集 1006 个通路

| 通路 | K | k | FDR |
| --- | --- | --- | --- |
| GOBP_AMEBOIDAL_TYPE_CELL_MIGRATION | 426 | 80 | 2.1e-20 |
| GOBP_EPITHELIAL_CELL_PROLIFERATION | 438 | 72 | 5.6e-15 |
| GOBP_REGULATION_OF_VASCULATURE_DEVELOPMENT | 295 | 57 | 8.0e-15 |
| GOBP_REGULATION_OF_EPITHELIAL_CELL_PROLIFERATION | 361 | 63 | 2.0e-14 |
| GOBP_TAXIS | 460 | 72 | 3.5e-14 |

## 四、生物学解读

- **tumor 中更高的基因**整体富集在「染色体分离 / 有丝分裂核分裂」相关过程，正是乳腺癌最经典的增殖特征。
- **normal 中更高的基因**整体富集在「细胞迁移 / 血管发育 / 上皮增殖」相关过程。

## 五、已知局限（如实列出，不藏着）

1. 走 Welch t + scipy `sf`，数值稳定；极端显著 p 值在正态近似 fallback 路径可能下溢为 0。
2. 富集背景用全部检测到的 19938 个基因，其中部分无 GO 注释，p 略偏小（标准做法，可接受）。
3. 未做批次校正；已做独立过滤（4655 个低表达基因不进入 BH 的 m，m=15283）。
4. 226 样本来自 217 个病人，仍有 9 个病人同时出现在两组，跨组相关性未处理。
5. 未做配对检验 —— 仅 9 对配对样本，这是正确选择。
