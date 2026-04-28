"""
医疗数据上的 HAN 实验入口（结构对齐原 ex_acm3025.py，复用原 HeteGAT_multi 模型不改动）。

支持三种 input mode（与 modeling_raw_plan.md §四对应）：
  --mode icd_only        : 节点输入 = ICD 向量；走原 HeteGAT_multi
  --mode icd_ukb_early   : 节点输入 = ICD || UKB 拼接；走原 HeteGAT_multi
  --mode icd_ukb_late    : 节点输入 = ICD；HeteGAT_multi → final_embed 后与 UKB 拼接 → 新增 dense 分类头
                           （原 HeteGAT_multi 内部代码不变；late fusion 在外部 graph 中完成）

最小改动清单（相对原 ex_acm3025.py）：
  1) load_data_dblp(.mat) → load_medical_data(.csv) （在 prepare_data.py 中）
  2) 新增 build_inputs() 处理三种 mode 的节点特征 / UKB 输入
  3) 新增 late fusion 路径：取 HeteGAT_multi 返回的 final_embed，拼接 UKB，过新 dense 头
  4) 评估指标改为 F1 / AUC / Acc / Precision（原版只输出 acc + KNN/Kmeans）
  5) 结果累加写入 results.csv
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np

# ===== TF1 / TF2 兼容 shim =====
# 原 repo 是 TF1，云端环境若装的是 TF2，这里把 `tensorflow` 重定向到 compat.v1，
# 这样下面 `from models import HeteGAT_multi`（其内部 `import tensorflow as tf`）也走 v1。
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    sys.modules["tensorflow"] = tf
except (ImportError, AttributeError):
    import tensorflow as tf  # 真 TF1 环境直接用

# numpy 新版本移除了 np.bool；原 repo 的 sample_mask / process.adj_to_bias 可能用到
if not hasattr(np, "bool"):
    np.bool = bool

# 把 HAN repo 根目录加入 sys.path，以便 import 原 repo 模块
_HERE = os.path.dirname(os.path.abspath(__file__))
_HAN_ROOT = os.path.dirname(_HERE)
if _HAN_ROOT not in sys.path:
    sys.path.insert(0, _HAN_ROOT)

from models import HeteGAT_multi          # 原 repo 模型，未做任何修改
from utils import process                  # 原 repo 工具，未做任何修改

# 适配代码
from prepare_data import load_medical_data

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
)


def sample_mask(idx, l):
    """与原 ex_acm3025.py 同名、同语义。"""
    m = np.zeros(l)
    m[idx] = 1
    return np.array(m, dtype=np.bool)


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


def train_and_eval(
    csv_path: str,
    mode: str,
    disease: str,
    *,
    batch_size: int = 1,
    nb_epochs: int = 200,
    patience: int = 30,
    lr: float = 0.005,
    l2_coef: float = 0.001,
    hid_units=(8,),
    n_heads=(8, 1),
    residual: bool = False,
    seed: int = 42,
) -> dict:
    """
    训练 + 评估单个 (csv, mode) 配置，返回 test 集上 best-val 时的指标 dict。
    """
    use_ukb = mode != "icd_only"
    data = load_medical_data(csv_path, use_ukb=use_ukb, seed=seed)

    node_input, ukb_for_late = build_inputs(data, mode)
    N = node_input.shape[0]
    F = node_input.shape[1]
    nb_classes = data["label"].shape[1]

    print(
        f"[{disease} | {mode}] N={N}, feat_dim={F}, "
        f"ukb_dim={ukb_for_late.shape[1] if ukb_for_late is not None else 0}, "
        f"meta_paths={data['meta_path_names']}, "
        f"cases/controls={data['n_cases']}/{data['n_controls']}"
    )

    # ===== mask & label split（沿用原 ex_acm3025.py 风格） =====
    y = data["label"]
    train_mask = sample_mask(data["train_idx"], N)
    val_mask = sample_mask(data["val_idx"], N)
    test_mask = sample_mask(data["test_idx"], N)
    y_train = np.zeros_like(y); y_train[train_mask] = y[train_mask]
    y_val = np.zeros_like(y);   y_val[val_mask] = y[val_mask]
    y_test = np.zeros_like(y);  y_test[test_mask] = y[test_mask]

    # ===== 邻接 → bias matrix（与原版完全一致） =====
    # 原版：rownetworks = [data['PAP'] - I, data['PLP'] - I]
    adj_list = [adj - np.eye(N, dtype=np.float32) for adj in data["adj_list"]]
    fea_list = [node_input for _ in range(len(adj_list))]

    # 加 batch 维（与原版一致）
    fea_list_b = [fea[np.newaxis] for fea in fea_list]
    adj_list_b = [adj[np.newaxis] for adj in adj_list]
    y_train = y_train[np.newaxis]; y_val = y_val[np.newaxis]; y_test = y_test[np.newaxis]
    train_mask_b = train_mask[np.newaxis]
    val_mask_b = val_mask[np.newaxis]
    test_mask_b = test_mask[np.newaxis]
    biases_list = [process.adj_to_bias(adj, [N], nhood=1) for adj in adj_list_b]

    # ===== 构图 =====
    tf.reset_default_graph()
    with tf.Graph().as_default():
        ftr_in_list = [
            tf.placeholder(tf.float32, shape=(batch_size, N, F), name=f"ftr_in_{i}")
            for i in range(len(fea_list_b))
        ]
        bias_in_list = [
            tf.placeholder(tf.float32, shape=(batch_size, N, N), name=f"bias_in_{i}")
            for i in range(len(biases_list))
        ]
        lbl_in = tf.placeholder(tf.int32, shape=(batch_size, N, nb_classes), name="lbl_in")
        msk_in = tf.placeholder(tf.int32, shape=(batch_size, N), name="msk_in")
        attn_drop = tf.placeholder(tf.float32, shape=(), name="attn_drop")
        ffd_drop = tf.placeholder(tf.float32, shape=(), name="ffd_drop")
        is_train = tf.placeholder(tf.bool, shape=(), name="is_train")

        # ===== 直接调用原 HeteGAT_multi.inference（未做任何修改） =====
        logits, final_embed, att_val = HeteGAT_multi.inference(
            ftr_in_list,
            nb_classes,
            N,
            is_train,
            attn_drop,
            ffd_drop,
            bias_mat_list=bias_in_list,
            hid_units=list(hid_units),
            n_heads=list(n_heads),
            residual=residual,
            activation=tf.nn.elu,
        )

        # ===== Late fusion：在原 inference 输出之上追加分类头（不改原模型） =====
        if mode == "icd_ukb_late":
            ukb_dim = ukb_for_late.shape[1]
            ukb_in = tf.placeholder(tf.float32, shape=(N, ukb_dim), name="ukb_in")
            fused = tf.concat([final_embed, ukb_in], axis=-1)            # (N, hidden*K + ukb_dim)
            logits_lf = tf.layers.dense(fused, nb_classes, activation=None,
                                        name="late_fusion_clf")
            logits_used = tf.expand_dims(logits_lf, axis=0)              # (1, N, C)
        else:
            ukb_in = None
            logits_used = logits                                         # (1, N, C)

        log_resh = tf.reshape(logits_used, [-1, nb_classes])
        lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])

        # 复用原 BaseGAttN 的 loss/优化器（HeteGAT_multi 继承自 BaseGAttN）
        loss = HeteGAT_multi.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        train_op = HeteGAT_multi.training(loss, lr, l2_coef)
        probs = tf.nn.softmax(log_resh)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True

        best_val_auc = -np.inf
        best_test_metrics = None
        wait = 0

        with tf.Session(config=cfg) as sess:
            sess.run(init_op)
            np.random.seed(seed)
            tf.set_random_seed(seed)

            base_fd = {ftr: d for ftr, d in zip(ftr_in_list, fea_list_b)}
            base_fd.update({b: a for b, a in zip(bias_in_list, biases_list)})

            for epoch in range(nb_epochs):
                fd = dict(base_fd)
                fd.update({lbl_in: y_train, msk_in: train_mask_b,
                           is_train: True, attn_drop: 0.6, ffd_drop: 0.6})
                if ukb_in is not None:
                    fd[ukb_in] = ukb_for_late
                _, loss_tr = sess.run([train_op, loss], feed_dict=fd)

                # ----- val -----
                fd_v = dict(base_fd)
                fd_v.update({lbl_in: y_val, msk_in: val_mask_b,
                             is_train: False, attn_drop: 0.0, ffd_drop: 0.0})
                if ukb_in is not None:
                    fd_v[ukb_in] = ukb_for_late
                p_val = sess.run(probs, feed_dict=fd_v)

                vmask = val_mask_b.reshape(-1).astype(bool)
                yv = y[data["val_idx"]].argmax(axis=1)
                p1_v = p_val[vmask, 1]
                auc_v = roc_auc_score(yv, p1_v) if len(np.unique(yv)) > 1 else float("nan")

                if not np.isnan(auc_v) and auc_v > best_val_auc:
                    best_val_auc = auc_v
                    # snapshot test metrics
                    fd_t = dict(base_fd)
                    fd_t.update({lbl_in: y_test, msk_in: test_mask_b,
                                 is_train: False, attn_drop: 0.0, ffd_drop: 0.0})
                    if ukb_in is not None:
                        fd_t[ukb_in] = ukb_for_late
                    p_te = sess.run(probs, feed_dict=fd_t)
                    tmask = test_mask_b.reshape(-1).astype(bool)
                    yt = y[data["test_idx"]].argmax(axis=1)
                    pred_te = p_te[tmask].argmax(axis=1)
                    p1_te = p_te[tmask, 1]
                    best_test_metrics = {
                        "f1": f1_score(yt, pred_te, zero_division=0),
                        "auc": roc_auc_score(yt, p1_te) if len(np.unique(yt)) > 1 else float("nan"),
                        "acc": accuracy_score(yt, pred_te),
                        "precision": precision_score(yt, pred_te, zero_division=0),
                    }
                    wait = 0
                else:
                    wait += 1

                if (epoch + 1) % 25 == 0 or epoch == 0:
                    print(f"  epoch {epoch+1:3d} | loss {loss_tr:.4f} | "
                          f"val auc {auc_v:.4f} | best {best_val_auc:.4f}")

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
    ap.add_argument("--disease", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    metrics = train_and_eval(
        csv_path=args.csv, mode=args.mode, disease=args.disease,
        nb_epochs=args.epochs, patience=args.patience, seed=args.seed,
    )
    row = {"model": "HAN", "Input": _INPUT_LABEL[args.mode],
           "disease": args.disease, **metrics}
    print("RESULT:", row)

    import pandas as pd
    df = pd.DataFrame([row], columns=["model", "Input", "disease",
                                       "f1", "auc", "acc", "precision"])
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"appended to {args.out}")


if __name__ == "__main__":
    main()
