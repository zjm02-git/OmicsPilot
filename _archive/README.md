# _archive｜归档区（不是死代码，是"历史版本"）

这些文件**不参与运行**，也不被任何代码 import。放在这里而不是删除，是因为它们记录了"改之前长什么样"，
面试或复盘时可能要用。

| 文件 | 原位置 | 为什么归档 |
|---|---|---|
| `OmicsAgent真实分析报告.md` | 项目根 | ⚠️ **结论错误的旧报告**：log2FC 对已取过对数的数据又取了一次 log2，且方向反了（把"在 normal 中更高"写成"在肿瘤中上调"）。保留作为 before/after 对照，**不要当结果引用** |
| `sample_report.md` | 项目根 | 更早的样例报告 |
| `backtrack_demo.py` | 项目根 | 回溯（时间旅行）演示脚本，主流程不依赖 |
| `verify_qa.py` | 项目根 | 早期手工验证：RAG 问答是否接通。现由 `tests/` + `tools/e2e_live.py` 覆盖 |
| `verify_rag.py` | 项目根 | 同上 |
| `rag/build_vector_db.py` | `rag/` | 往 chromadb 写向量，且写死绝对路径。现链路改用 `data/vecs.npy`（numpy 暴力全扫） |
| `data/fix_vector_db.py` | `data/` | 修复 chromadb HNSW 索引的临时脚本，同样已弃用 |
| `data/retrieve.py` | `data/` | 早期检索原型，能力已并入 `omics_agent/rag.py` |

## 为什么它们可以安全移走

已确认：`chromadb` 只被 `rag/build_vector_db.py` 和 `data/fix_vector_db.py` 两个文件 import，
真实链路（`rag.py` → `vecs.npy`）完全绕开 chromadb。
移走后全量 80 个 unittest 仍然全绿。
