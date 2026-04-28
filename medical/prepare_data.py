"""
医疗 CSV → HAN 输入张量。

输出字典字段对齐原 ex_acm3025.py / load_data_dblp 的 .mat 结构：
  feature       : (N, F_icd) float32     —— 节点特征（ICD 向量）
  label         : (N, 2)     float32     —— one-hot
  adj_list      : List[(N, N) float32]   —— 各 meta-path 邻接（含自环）
  meta_path_names: List[str]             —— 与 adj_list 一一对应（如 'PIP_I'）
  train_idx / val_idx / test_idx: 1D index arrays
  ukb_features  : (N, F_ukb) float32 或 None  —— 仅 use_ukb=True 时返回

每个病人取 1 行：case 优先 x_row（无标签泄漏），control 用 control 行。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# 与目标疾病最相关的 ICD-10 章节前缀，每个章节定义一条 PIP meta-path
DEFAULT_META_PATH_CHAPTERS: List[str] = ["I", "E", "M", "K"]
META_COLS = {"eid", "event_dt", "y_label", "record_type"}


def _identify_columns(df: pd.DataFrame):
    """ICD 列：大写字母开头；UKB 列：数字开头（与 modeling_raw_plan.md §三 定义一致）。"""
    icd_cols, ukb_cols = [], []
    for c in df.columns:
        cs = str(c)
        if c in META_COLS or cs.startswith("Unnamed"):
            continue
        if re.match(r"^[A-Z]", cs):
            icd_cols.append(c)
        elif re.match(r"^\d", cs):
            ukb_cols.append(c)
    return icd_cols, ukb_cols


def _select_feature_row(group: pd.DataFrame) -> pd.Series:
    """case 优先 x_row（避免 y_row 当日诊断的标签泄漏）；control 用唯一 control 行。"""
    rt = group["record_type"].values
    if "x_row" in rt:
        return group[group["record_type"] == "x_row"].iloc[0]
    if "control" in rt:
        return group[group["record_type"] == "control"].iloc[0]
    return group.iloc[0]


def _build_pip_adj(sub_matrix: np.ndarray, num_nodes: int) -> np.ndarray:
    """
    给定 病人×ICD-子集 二值矩阵，构造 PIP 邻接（含自环，与原 ex_acm3025.py 风格一致）：
        A_PIP = (M @ M.T > 0) + I_N
    """
    if sub_matrix.shape[1] == 0:
        return np.eye(num_nodes, dtype=np.float32)
    coo = sub_matrix @ sub_matrix.T
    np.fill_diagonal(coo, 0)
    adj = (coo > 0).astype(np.float32)
    adj += np.eye(num_nodes, dtype=np.float32)
    return adj


def load_medical_data(
    csv_path: str | Path,
    use_ukb: bool = False,
    meta_path_chapters: Optional[List[str]] = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    csv_path = Path(csv_path)
    if meta_path_chapters is None:
        meta_path_chapters = DEFAULT_META_PATH_CHAPTERS

    df = pd.read_csv(csv_path)
    icd_cols, ukb_cols = _identify_columns(df)

    one = (
        df.groupby("eid", sort=False, group_keys=False).apply(_select_feature_row)
        .reset_index(drop=True)
    )

    eids = one["eid"].values
    labels = one["y_label"].astype(int).values
    icd = one[icd_cols].astype(np.float32).values
    N = len(eids)

    adj_list: List[np.ndarray] = []
    used: List[str] = []
    for ch in meta_path_chapters:
        sub_cols = [c for c in icd_cols if str(c).startswith(ch)]
        if not sub_cols:
            continue
        sub = (one[sub_cols].astype(np.float32).values > 0).astype(np.int8)
        adj = _build_pip_adj(sub, N)
        adj_list.append(adj)
        used.append(f"PIP_{ch}")

    if len(adj_list) < 2:
        raise RuntimeError(
            f"only {len(adj_list)} usable meta-paths; HAN needs >= 2"
        )

    # 二分类 one-hot 标签
    y = np.zeros((N, 2), dtype=np.float32)
    y[np.arange(N), labels] = 1.0

    idx = np.arange(N)
    train_idx, temp_idx = train_test_split(
        idx, train_size=train_ratio, stratify=labels, random_state=seed
    )
    val_size = val_ratio / (1.0 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_size, stratify=labels[temp_idx], random_state=seed
    )

    ukb_features = None
    if use_ukb and ukb_cols:
        ukb = one[ukb_cols].astype(np.float32).values
        ukb = np.nan_to_num(ukb, nan=0.0, posinf=0.0, neginf=0.0)
        std = ukb.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        ukb = (ukb - ukb.mean(axis=0, keepdims=True)) / std
        ukb_features = ukb.astype(np.float32)

    return {
        "feature": icd,
        "label": y,
        "labels_int": labels,
        "adj_list": adj_list,
        "meta_path_names": used,
        "train_idx": np.asarray(train_idx),
        "val_idx": np.asarray(val_idx),
        "test_idx": np.asarray(test_idx),
        "ukb_features": ukb_features,
        "n_cases": int(labels.sum()),
        "n_controls": int((labels == 0).sum()),
        "icd_cols": icd_cols,
        "ukb_cols": ukb_cols,
    }
