"""
HeteGAT_multi 的 PyTorch 端口 —— 与原 repo 中 models/gat.py + utils/layers.py 一一对应。

模块到原文件的映射:
  AttnHead         <- utils.layers.attn_head        (TF1)
  SimpleAttLayer   <- utils.layers.SimpleAttLayer   (TF1)
  HeteGATMulti     <- models.gat.HeteGAT_multi      (TF1)

数学等价（每行注释指向 TF1 原码行号或片段）：
  - 1×1 conv1d 在 TF 中等价于在最后一维上的 Linear（不引入 spatial 维度操作），
    PyTorch 用 nn.Linear 实现完全等价。
  - bias_mat: 与原 utils.process.adj_to_bias 输出一致（边为 0、非边为 -1e9，softmax 屏蔽）。

云端环境前提: 仅需 torch（已普遍预装），不依赖 TF / DGL。
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Node-level attention head（对应 utils/layers.py:attn_head）
# ---------------------------------------------------------------------------
class AttnHead(nn.Module):
    """
    单头 GAT attention（dense 邻接 + bias-mat 屏蔽），对应原 attn_head 函数。

    原 TF 实现要点（utils/layers.py L7-L46）:
        seq_fts = conv1d(seq, out_sz, 1, use_bias=False)    # 共享投影 W
        f_1     = conv1d(seq_fts, 1, 1)                     # 注意力打分 a^T
        f_2     = conv1d(seq_fts, 1, 1)
        logits  = f_1 + f_2^T                               # 加性 attention
        coefs   = softmax(LeakyReLU(logits) + bias_mat)
        vals    = coefs @ seq_fts + bias
    """

    def __init__(
        self,
        in_sz: int,
        out_sz: int,
        in_drop: float = 0.0,
        coef_drop: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        self.W = nn.Linear(in_sz, out_sz, bias=False)   # 等价 conv1d(seq, out_sz, 1, use_bias=False)
        self.a1 = nn.Linear(out_sz, 1)                  # 等价 conv1d(seq_fts, 1, 1)
        self.a2 = nn.Linear(out_sz, 1)                  # 等价 conv1d(seq_fts, 1, 1)
        self.bias = nn.Parameter(torch.zeros(out_sz))   # 等价 tf.contrib.layers.bias_add
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.res_fc = nn.Linear(in_sz, out_sz, bias=False) if (residual and in_sz != out_sz) else None

    def forward(self, seq: torch.Tensor, bias_mat: torch.Tensor) -> torch.Tensor:
        # seq:      (B, N, in_sz)
        # bias_mat: (B, N, N)，边=0，非边=-1e9
        if self.in_drop > 0.0 and self.training:
            seq = F.dropout(seq, p=self.in_drop)
        seq_fts = self.W(seq)                          # (B, N, out_sz)

        f1 = self.a1(seq_fts)                          # (B, N, 1)
        f2 = self.a2(seq_fts)                          # (B, N, 1)
        logits = f1 + f2.transpose(1, 2)               # (B, N, N)
        coefs = F.softmax(F.leaky_relu(logits) + bias_mat, dim=-1)

        if self.coef_drop > 0.0 and self.training:
            coefs = F.dropout(coefs, p=self.coef_drop)
        if self.in_drop > 0.0 and self.training:
            seq_fts = F.dropout(seq_fts, p=self.in_drop)

        vals = torch.matmul(coefs, seq_fts) + self.bias

        if self.residual:
            if self.res_fc is not None:
                vals = vals + self.res_fc(seq)
            else:
                vals = vals + seq
        return vals


# ---------------------------------------------------------------------------
# Semantic-level attention（对应 utils/layers.py:SimpleAttLayer）
# ---------------------------------------------------------------------------
class SimpleAttLayer(nn.Module):
    """
    Meta-path 维度 attention：对每个 meta-path 嵌入打分加权聚合，对应原 SimpleAttLayer。

    原 TF 实现:
        v       = tanh(inputs @ W + b)
        vu      = v @ u
        alphas  = softmax(vu)
        output  = sum_t (inputs[:, t] * alphas[:, t])
    """

    def __init__(self, in_sz: int, attention_size: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_sz, attention_size) * 0.1)
        self.b = nn.Parameter(torch.randn(attention_size) * 0.1)
        self.u = nn.Parameter(torch.randn(attention_size) * 0.1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs: (N_or_B, T, D)，T = num meta-paths, D = embed_dim
        v = torch.tanh(torch.matmul(inputs, self.W) + self.b)   # (..., T, A)
        vu = torch.matmul(v, self.u)                             # (..., T)
        alphas = F.softmax(vu, dim=-1)                           # (..., T)
        output = (inputs * alphas.unsqueeze(-1)).sum(dim=-2)     # (..., D)
        return output, alphas


# ---------------------------------------------------------------------------
# HeteGAT_multi 主体（对应 models/gat.py:HeteGAT_multi.inference）
# ---------------------------------------------------------------------------
class HeteGATMulti(nn.Module):
    """
    论文配置 (hid_units=[8], n_heads=[8, 1]) 时的等价计算流：
        for each meta-path:
            8 heads × AttnHead(in_sz, 8) → concat → (N, 64)
        stack 4 meta-paths → (N, 4, 64)
        SimpleAttLayer → (N, 64)
        n_heads[-1]=1 个 Linear(64 → nb_classes)，多个时取平均
    """

    def __init__(
        self,
        num_meta_paths: int,
        in_sz: int,
        hid_units: List[int],
        n_heads: List[int],
        nb_classes: int,
        mp_att_size: int = 128,
        in_drop: float = 0.6,
        coef_drop: float = 0.6,
        residual: bool = False,
        activation=F.elu,
    ):
        super().__init__()
        assert len(n_heads) == len(hid_units) + 1, \
            "原 ex_acm3025.py 约定: len(n_heads) = len(hid_units) + 1"
        self.num_meta_paths = num_meta_paths
        self.activation = activation

        # 第一层 attention：每个 meta-path 各 n_heads[0] 个头
        self.layer0 = nn.ModuleList()
        for _ in range(num_meta_paths):
            heads = nn.ModuleList([
                AttnHead(in_sz, hid_units[0], in_drop, coef_drop, residual=False)
                for _ in range(n_heads[0])
            ])
            self.layer0.append(heads)

        # 中间层（hid_units 长度 > 1 时启用）
        self.intermediate_layers = nn.ModuleList()
        for li in range(1, len(hid_units)):
            this_layer = nn.ModuleList()
            in_size_li = hid_units[li - 1] * n_heads[li - 1]
            for _ in range(num_meta_paths):
                heads = nn.ModuleList([
                    AttnHead(in_size_li, hid_units[li], in_drop, coef_drop, residual=residual)
                    for _ in range(n_heads[li])
                ])
                this_layer.append(heads)
            self.intermediate_layers.append(this_layer)

        # 各 meta-path 嵌入维度（concat 多头之后）
        if len(hid_units) == 1:
            self.embed_dim = hid_units[0] * n_heads[0]
        else:
            self.embed_dim = hid_units[-1] * n_heads[len(hid_units) - 1]

        # Meta-path 维度的 semantic attention
        self.semantic_attn = SimpleAttLayer(self.embed_dim, mp_att_size)

        # 最终分类头：n_heads[-1] 个 Linear，输出取平均（对应原代码 add_n / n_heads[-1]）
        self.n_final_heads = n_heads[-1]
        self.classifiers = nn.ModuleList([
            nn.Linear(self.embed_dim, nb_classes) for _ in range(n_heads[-1])
        ])

    # -----------------------------------------------------------------------
    # 拆分 encode / forward 以支持 late fusion 在外部追加分类头
    # -----------------------------------------------------------------------
    def encode(
        self,
        inputs_list: List[torch.Tensor],
        bias_mat_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """每条 meta-path 跑一遍 attention 栈，再过 SimpleAttLayer 聚合。"""
        embed_per_mp = []
        for mp_idx in range(self.num_meta_paths):
            seq = inputs_list[mp_idx]                  # (B, N, F)
            bias_mat = bias_mat_list[mp_idx]           # (B, N, N)

            head_outs = [self.activation(h(seq, bias_mat)) for h in self.layer0[mp_idx]]
            h = torch.cat(head_outs, dim=-1)           # (B, N, hid*K)

            for layer in self.intermediate_layers:
                head_outs = [self.activation(h_i(h, bias_mat)) for h_i in layer[mp_idx]]
                h = torch.cat(head_outs, dim=-1)

            # squeeze batch (B=1 与原版一致)，扩 meta-path 维 → (N, 1, embed_dim)
            embed_per_mp.append(h.squeeze(0).unsqueeze(1))

        multi_embed = torch.cat(embed_per_mp, dim=1)        # (N, M, embed_dim)
        final_embed, att_val = self.semantic_attn(multi_embed)   # (N, embed_dim), (N, M)
        return final_embed, att_val

    def forward(
        self,
        inputs_list: List[torch.Tensor],
        bias_mat_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        final_embed, att_val = self.encode(inputs_list, bias_mat_list)
        outs = [clf(final_embed) for clf in self.classifiers]
        logits = torch.stack(outs, dim=0).mean(dim=0)        # 与原 add_n / n_heads[-1] 一致
        return logits, final_embed, att_val


# ---------------------------------------------------------------------------
# Late fusion 变体：在 final_embed 之后拼 UKB，再走分类头
# ---------------------------------------------------------------------------
class HeteGATMultiLateFusion(HeteGATMulti):
    """
    与 HeteGATMulti 的唯一差别在分类头维度（embed_dim + ukb_dim → nb_classes）。
    encode() 完全继承，原模型主体不动。
    """

    def __init__(self, *args, ukb_dim: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.ukb_dim = ukb_dim
        # 覆盖 classifiers：输入维 = embed_dim + ukb_dim
        self.classifiers = nn.ModuleList([
            nn.Linear(self.embed_dim + ukb_dim, self.classifiers[0].out_features)
            for _ in range(self.n_final_heads)
        ])

    def forward(
        self,
        inputs_list: List[torch.Tensor],
        bias_mat_list: List[torch.Tensor],
        ukb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        final_embed, att_val = self.encode(inputs_list, bias_mat_list)   # (N, embed_dim)
        fused = torch.cat([final_embed, ukb], dim=-1)                    # (N, embed_dim + ukb_dim)
        outs = [clf(fused) for clf in self.classifiers]
        logits = torch.stack(outs, dim=0).mean(dim=0)
        return logits, fused, att_val


# ---------------------------------------------------------------------------
# 邻接 → bias matrix（对应 utils.process.adj_to_bias 在 nhood=1 下的等价行为）
# ---------------------------------------------------------------------------
def adj_to_bias(adj: torch.Tensor, neg_inf: float = -1e9) -> torch.Tensor:
    """
    输入: 二值邻接 (B, N, N)（含自环）
    输出: bias_mat：边处=0，非边处=neg_inf
    与原 utils/process.py:adj_to_bias 在 nhood=1 时等价。
    """
    return torch.where(adj > 0, torch.zeros_like(adj), torch.full_like(adj, neg_inf))


__all__ = [
    "AttnHead",
    "SimpleAttLayer",
    "HeteGATMulti",
    "HeteGATMultiLateFusion",
    "adj_to_bias",
]
