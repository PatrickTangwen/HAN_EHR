"""
医疗数据上的 HAN 实验入口（PyTorch 版本，结构与原 ex_acm3025.py 对齐）。

为什么是 PyTorch:
  原 Jhy1993/HAN 是 TF1 代码，云端环境未必装 TF。本文件改用 han_pytorch.py 中的
  HeteGATMulti / HeteGATMultiLateFusion —— 这两个类与原 models/gat.py 的 HeteGAT_multi
  逐函数对应翻译，不是简化版重写。

支持三种 input mode（与 modeling_raw_plan.md §四对应）:
  --mode icd_only        节点输入 = ICD 向量；HeteGATMulti
  --mode icd_ukb_early   节点输入 = ICD || UKB；HeteGATMulti
  --mode icd_ukb_late    节点输入 = ICD；HeteGATMulti.encode → 拼 UKB → 新分类头
                          （在 HeteGATMultiLateFusion 中实现，原 encode 不动）

最小改动相对原 ex_acm3025.py：
  1) load_data_dblp(.mat) → load_medical_data(.csv)
  2) build_inputs() 处理三种 mode
  3) late fusion 在外部分类头实现（不改 encode）
  4) 评估指标改为 F1 / AUC / Acc / Precision
  5) 结果累加写入 results.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# 让 `from prepare_data import ...` 在任意 CWD 下都能找到模块
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from prepare_data import load_medical_data
from han_pytorch import HeteGATMulti, HeteGATMultiLateFusion, adj_to_bias

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_inputs(data: dict, mode: str):
    """根据 input mode 决定 HeteGAT 节点特征与 UKB 注入张量。"""
    feat_icd = data["feature"]
    ukb = data["ukb_features"]
    if mode == "icd_only":
        return feat_icd.astype(np.float32), None
    if mode == "icd_ukb_early":
        if ukb is None:
            raise ValueError("icd_ukb_early needs use_ukb=True data")
        return np.concatenate([feat_icd, ukb], axis=1).astype(np.float32), None
    if mode == "icd_ukb_late":
        if ukb is None:
            raise ValueError("icd_ukb_late needs use_ukb=True data")
        return feat_icd.astype(np.float32), ukb.astype(np.float32)
    raise ValueError(f"unknown mode: {mode}")


def evaluate_metrics(probs_np: np.ndarray, labels_np: np.ndarray) -> dict:
    """二分类: F1 / AUC / Acc / Precision（正类）。"""
    preds = probs_np.argmax(axis=1)
    p1 = probs_np[:, 1]
    auc = roc_auc_score(labels_np, p1) if len(np.unique(labels_np)) > 1 else float("nan")
    return {
        "f1": f1_score(labels_np, preds, zero_division=0),
        "auc": auc,
        "acc": accuracy_score(labels_np, preds),
        "precision": precision_score(labels_np, preds, zero_division=0),
    }


def train_and_eval(
    csv_path: str,
    mode: str,
    disease: str,
    *,
    format: str = "aggregated",
    nb_epochs: int = 200,
    patience: int = 30,
    lr: float = 0.005,
    l2_coef: float = 0.001,
    hid_units=(8,),
    n_heads=(8, 1),
    in_drop: float = 0.6,
    coef_drop: float = 0.6,
    residual: bool = False,
    seed: int = 42,
    device: str = None,
) -> dict:
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    use_ukb = mode != "icd_only"
    data = load_medical_data(csv_path, use_ukb=use_ukb, format=format, seed=seed)

    node_input_np, ukb_for_late_np = build_inputs(data, mode)
    N = node_input_np.shape[0]
    F_in = node_input_np.shape[1]
    nb_classes = data["label"].shape[1]
    labels_int = data["labels_int"]
    n_meta_paths = len(data["adj_list"])

    print(
        f"[{disease} | {format} | {mode}] N={N}, feat_dim={F_in}, "
        f"ukb_dim={ukb_for_late_np.shape[1] if ukb_for_late_np is not None else 0}, "
        f"meta_paths={data['meta_path_names']}, "
        f"cases/controls={data['n_cases']}/{data['n_controls']}, device={dev}"
    )

    # === 构造 bias_mat：边=0, 非边=-1e9（与原 utils.process.adj_to_bias 一致） ===
    # data["adj_list"] 已含自环
    adj_stack = torch.from_numpy(np.stack(data["adj_list"], axis=0)).float()    # (M, N, N)
    bias_mat_stack = adj_to_bias(adj_stack)                                      # (M, N, N)

    # 加 batch 维并搬 device
    node_input = torch.from_numpy(node_input_np).float().unsqueeze(0).to(dev)    # (1, N, F)
    inputs_list = [node_input for _ in range(n_meta_paths)]                      # 与原 ex_acm3025.py 一致：3 份相同特征
    bias_mat_list = [bias_mat_stack[i].unsqueeze(0).to(dev) for i in range(n_meta_paths)]
    labels_t = torch.from_numpy(labels_int).long().to(dev)                       # (N,)
    if ukb_for_late_np is not None:
        ukb_t = torch.from_numpy(ukb_for_late_np).float().to(dev)
    else:
        ukb_t = None

    train_idx = torch.from_numpy(data["train_idx"]).long().to(dev)
    val_idx = torch.from_numpy(data["val_idx"]).long().to(dev)
    test_idx = torch.from_numpy(data["test_idx"]).long().to(dev)

    # === 模型 ===
    if mode == "icd_ukb_late":
        model = HeteGATMultiLateFusion(
            num_meta_paths=n_meta_paths,
            in_sz=F_in,
            hid_units=list(hid_units),
            n_heads=list(n_heads),
            nb_classes=nb_classes,
            ukb_dim=ukb_for_late_np.shape[1],
            in_drop=in_drop,
            coef_drop=coef_drop,
            residual=residual,
        ).to(dev)
    else:
        model = HeteGATMulti(
            num_meta_paths=n_meta_paths,
            in_sz=F_in,
            hid_units=list(hid_units),
            n_heads=list(n_heads),
            nb_classes=nb_classes,
            in_drop=in_drop,
            coef_drop=coef_drop,
            residual=residual,
        ).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    # 类别权重缓解 case/control 不平衡
    n_pos = max(int((labels_int == 1).sum()), 1)
    n_neg = max(int((labels_int == 0).sum()), 1)
    cls_weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32, device=dev)

    def _forward():
        if mode == "icd_ukb_late":
            logits, _, _ = model(inputs_list, bias_mat_list, ukb_t)
        else:
            logits, _, _ = model(inputs_list, bias_mat_list)
        return logits

    best_val_auc = -np.inf
    best_test_metrics = None
    wait = 0

    for epoch in range(nb_epochs):
        model.train()
        logits = _forward()
        loss = F.cross_entropy(logits[train_idx], labels_t[train_idx], weight=cls_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- val ----
        model.eval()
        with torch.no_grad():
            logits_eval = _forward()
            probs_eval = F.softmax(logits_eval, dim=-1).cpu().numpy()

        val_metrics = evaluate_metrics(probs_eval[val_idx.cpu().numpy()],
                                       labels_int[data["val_idx"]])
        if not np.isnan(val_metrics["auc"]) and val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_test_metrics = evaluate_metrics(probs_eval[test_idx.cpu().numpy()],
                                                 labels_int[data["test_idx"]])
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d} | loss {loss.item():.4f} | "
                  f"val auc {val_metrics['auc']:.4f} | best {best_val_auc:.4f}")

        if wait >= patience:
            print(f"  early stop at epoch {epoch+1}")
            break

    if best_test_metrics is None:
        best_test_metrics = {"f1": float("nan"), "auc": float("nan"),
                             "acc": float("nan"), "precision": float("nan")}
    return best_test_metrics


_INPUT_LABEL = {
    "icd_only": "icd only",
    "icd_ukb_early": "icd + ukb (early)",
    "icd_ukb_late": "icd + ukb (late)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["icd_only", "icd_ukb_early", "icd_ukb_late"])
    ap.add_argument("--format", default="aggregated",
                    choices=["aggregated", "longitudinal"],
                    help="数据格式；longitudinal 在 load 时按 record_type "
                         "过滤后做 OR 聚合（详见 prepare_data.py）")
    ap.add_argument("--disease", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None,
                    help="cuda / cpu；默认自动选择")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    metrics = train_and_eval(
        csv_path=args.csv, mode=args.mode, disease=args.disease,
        format=args.format,
        nb_epochs=args.epochs, patience=args.patience, seed=args.seed,
        device=args.device,
    )
    row = {"model": "HAN", "format": args.format,
           "Input": _INPUT_LABEL[args.mode],
           "disease": args.disease, **metrics}
    print("RESULT:", row)

    import pandas as pd
    df = pd.DataFrame([row], columns=["model", "format", "Input", "disease",
                                       "f1", "auc", "acc", "precision"])
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"appended to {args.out}")


if __name__ == "__main__":
    main()
