# HAN-EHR

把 Heterogeneous Graph Attention Network (HAN, WWW-2019) 适配到 UK Biobank EHR 数据上的疾病分类任务。

* 上游论文 / 原 repo：https://github.com/Jhy1993/Representation-Learning-on-Heterogeneous-Graph
* 原 HAN 是 TF1 实现，本仓库 `medical/` 提供与原 `models/gat.py` + `utils/layers.py` 逐函数对应的 PyTorch 端口（`han_pytorch.py`），未做简化。

---

## 仓库结构

```
HAN_EHR/
├── medical/                ← 医疗适配主目录（推荐入口）
│   ├── ex_medical.py       单个 (disease, format, mode) 实验入口
│   ├── run_experiments.py  批量 wrapper：disease × format × mode
│   ├── prepare_data.py     CSV → HAN 输入张量
│   ├── han_pytorch.py      HAN 模型（PyTorch 端口）
│   └── requirements.txt
│
├── models/                 ← 原 HAN repo 的 TF1 GAT 代码（参考用，不在 medical 流水线里调用）
├── utils/                  ← 原 HAN repo 的 TF1 工具函数（参考用）
├── ex_acm3025.py           ← 原 HAN 在 ACM3025 上的入口（参考用）
├── preprocess_dblp.py      ← 原 HAN 的 DBLP 预处理脚本（参考用）
├── data/                   ← 原 HAN 的样例数据（ACM、DBLP、IMDB；非医疗数据）
└── han.pdf                 ← 原 HAN 论文
```

> 训练医疗数据时只需进入 `medical/`，不要改动其他文件。`models/` 和 `utils/` 是 TF1 代码，仅作算法对照参考。

---

## 实验设计

支持 **2 种数据格式 × 3 种 input 配置** 共 6 个组合，每个疾病分别评估：

| 维度 | 取值 | 数据来源 |
|---|---|---|
| `format` | `aggregated` | `datasets_agg/` 或 `datasets_agg_no_ukb/` |
| `format` | `longitudinal` | `datasets_longitudinal/` 或 `datasets_longitudinal_no_ukb/` |
| `mode` | `icd_only` | `*_no_ukb/` 文件夹的 CSV |
| `mode` | `icd_ukb_early` | `*/` 文件夹的 CSV，UKB 在输入端拼入 ICD |
| `mode` | `icd_ukb_late` | `*/` 文件夹的 CSV，UKB 在分类头前拼接 |

HAN 是静态异质图模型；longitudinal 数据在 `prepare_data.py` 内按 `record_type` 过滤掉 `y_row` 后做 OR 聚合（ICD 取 max、UKB 取 first），节点仍是病人粒度。详见 `prepare_data.py` 的 `_build_per_patient_table`。

支持的 7 个疾病：

```
CKD  Cardiac_Fibrosis  Crohns_Disease  Fibrosis_of_Skin
MASH  Pulmonary_fibrosis  SSc_Connective_Tissue
```

---

## 数据目录约定

`run_experiments.py --data-root <root>` 期望 4 个并列子文件夹：

```
<root>/
├── datasets_agg/                     aggregated + UKB
├── datasets_agg_no_ukb/              aggregated, ICD only
├── datasets_longitudinal/            longitudinal + UKB
└── datasets_longitudinal_no_ukb/     longitudinal, ICD only
```

每个文件夹下文件命名统一为 `dataset_<Disease>.csv`。CSV 列约定：

* `eid`、`event_dt`、`y_label`、`record_type` 为 meta 列
* 大写字母开头的列（如 `I10`、`E11`）为 ICD 特征（约 1786 列）
* 数字开头的列（如 `30000`、`21001`）为 UKB 静态特征（约 204 列，仅 `*_with_ukb` 文件存在）

---

## 快速运行

环境要求：PyTorch ≥ 1.13、numpy、pandas、scikit-learn。GPU 推荐 A100 80GB（dense HAN 在 N≈14k 时显存压力大）。

### 单个配置（debug）

```bash
cd medical/
python ex_medical.py \
  --csv ../../datasets_agg_no_ukb/dataset_Cardiac_Fibrosis.csv \
  --mode icd_only \
  --format aggregated \
  --disease Cardiac_Fibrosis \
  --epochs 1 --patience 1 \
  --out debug_results.csv
```

> `--csv` 路径必须与 `--format` 匹配（aggregated 路径配 `aggregated`，longitudinal 路径配 `longitudinal`）。

### 单个疾病、全部 6 个配置

```bash
cd medical/
python run_experiments.py \
  --data-root ../.. \
  --diseases Cardiac_Fibrosis \
  --reset-out --out results.csv
```

### 全部 7 个疾病（42 个配置）

```bash
cd medical/
python run_experiments.py \
  --data-root ../.. \
  --diseases all \
  --reset-out --out results.csv
```

### 限定 format 或 mode

```bash
# 只跑 aggregated
python run_experiments.py --data-root ../.. --diseases CKD --formats aggregated

# 只跑 longitudinal + icd_only
python run_experiments.py --data-root ../.. --diseases CKD \
  --formats longitudinal --modes icd_only
```

---

## 输出格式

`results.csv` 列：`model, format, Input, disease, f1, auc, acc, precision`

* `model`：固定 `HAN`
* `format`：`aggregated` / `longitudinal`
* `Input`：`icd only` / `icd + ukb (early)` / `icd + ukb (late)`
* 指标：测试集上的 F1（正类）/ ROC-AUC / Accuracy / Precision（正类）

每条 `(model, format, Input, disease)` 一行。单个疾病默认输出 6 行，全部疾病 42 行。

---

## CLI 参数速览

### `ex_medical.py`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--csv` | 必填 | 输入 CSV 路径 |
| `--mode` | 必填 | `icd_only` / `icd_ukb_early` / `icd_ukb_late` |
| `--format` | `aggregated` | `aggregated` / `longitudinal` |
| `--disease` | 必填 | 疾病名（写入结果列） |
| `--epochs` | 200 | 最大训练轮数 |
| `--patience` | 30 | early stopping 耐心 |
| `--seed` | 42 | 随机种子 |
| `--device` | 自动 | `cuda` / `cpu` |
| `--out` | `results.csv` | 输出 CSV（追加写） |

### `run_experiments.py`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data-root` | 必填 | 含 4 个 dataset 子文件夹的根目录 |
| `--diseases` | 必填 | 疾病名列表，`all` 表示全部 7 个 |
| `--formats` | `aggregated longitudinal` | 数据格式列表 |
| `--modes` | 全部 3 种 | input 配置列表 |
| `--epochs` | 200 | 最大训练轮数 |
| `--patience` | 30 | early stopping 耐心 |
| `--seed` | 42 | 随机种子 |
| `--reset-out` | False | 运行前清空 `--out` |
| `--out` | `results.csv` | 输出 CSV |

---

## 关键实现说明

* **数据加载** (`prepare_data.py`)：`load_medical_data(csv_path, use_ukb, format, ...)` 返回 `feature, label, adj_list, meta_path_names, train/val/test_idx, ukb_features` 等字段，结构对齐原 `load_data_dblp` 的 `.mat` 输出。
* **Meta-path 构造**：按 ICD-10 章节前缀（默认 `I, E, M, K`）切分 ICD 列，每个章节构造一条 PIP 邻接 `A = (M @ M.T > 0) + I_N`，含自环。HAN 至少需要 2 条 meta-path。
* **Late fusion** (`han_pytorch.HeteGATMultiLateFusion`)：在不修改原 `encode` 的前提下，在分类头前拼接 UKB 静态特征——避免 longitudinal 中每个时间步重复输入静态特征的冗余。
* **类别不均衡**：交叉熵用 `class_weight = [1, n_neg/n_pos]` 缓解。
* **early stopping**：以 val AUC 为准，`patience` 轮无提升则停止。

---

## 在 Grace HPC 上运行

完整的 module load、venv 创建、salloc/sbatch 流程详见仓库外的 [`HAN_EHR_run_guide.md`](../HAN_EHR_run_guide.md)，包含：

* PyTorch module + venv 配置
* `gpu_devel` / `gpu` partition 申请
* salloc 交互式调试与 sbatch 批量提交
* 常见错误（OOM、CUDA 不可用、conda 冲突等）排查

---

## 引用

如使用 HAN 模型，请引用原论文：

```bibtex
@article{han2019,
  title={Heterogeneous Graph Attention Network},
  author={Wang, Xiao and Ji, Houye and Shi, Chuan and Wang, Bai and Cui, Peng and Yu, P. and Ye, Yanfang},
  journal={WWW},
  year={2019}
}
```

原 repo：https://github.com/Jhy1993/HAN
DGL 实现参考：https://github.com/dmlc/dgl/tree/master/examples/pytorch/han
