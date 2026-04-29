"""
对一个或多个疾病一次性跑全部 (format × input) 配置，结果累加写入 results.csv：

  HAN | aggregated   | icd only          | <disease>
  HAN | aggregated   | icd + ukb (early) | <disease>
  HAN | aggregated   | icd + ukb (late)  | <disease>
  HAN | longitudinal | icd only          | <disease>
  HAN | longitudinal | icd + ukb (early) | <disease>
  HAN | longitudinal | icd + ukb (late)  | <disease>

按 modeling_raw_plan_adjusted.md §一/§四.2，aggregated 与 longitudinal 两种格式必须都跑。
HAN 是静态异质图模型；longitudinal 在 load 时按 record_type 过滤掉 y_row 后做 OR 聚合
（详见 prepare_data.py），与 aggregated 数据流对齐。

约定的数据目录布局（与 modeling_raw_plan_adjusted.md §三 的 4 文件夹结构一致）:
  <data-root>/
    datasets_agg/                    dataset_{Disease}.csv   ← aggregated + UKB
    datasets_agg_no_ukb/             dataset_{Disease}.csv   ← aggregated, ICD only
    datasets_longitudinal/           dataset_{Disease}.csv   ← longitudinal + UKB
    datasets_longitudinal_no_ukb/    dataset_{Disease}.csv   ← longitudinal, ICD only

每个 (disease, format, mode) 配置开独立子进程运行，避免训练状态串扰。

用法（默认 2 种 format × 3 种 input 都跑）:
  python run_experiments.py --data-root /path/to/data --diseases Cardiac_Fibrosis

用法（全部 7 个疾病）:
  python run_experiments.py --data-root /path/to/data --diseases all --reset-out

用法（只跑 aggregated）:
  python run_experiments.py --data-root /path/to/data --diseases CKD --formats aggregated
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


ALL_DISEASES = [
    "CKD",
    "Cardiac_Fibrosis",
    "Crohns_Disease",
    "Fibrosis_of_Skin",
    "MASH",
    "Pulmonary_fibrosis",
    "SSc_Connective_Tissue",
]

# (format, with_ukb) → 子目录名（与用户实际目录命名对齐）
SUBDIR_MAP = {
    ("aggregated",   False): "datasets_agg_no_ukb",
    ("aggregated",   True):  "datasets_agg",
    ("longitudinal", False): "datasets_longitudinal_no_ukb",
    ("longitudinal", True):  "datasets_longitudinal",
}

ALL_FORMATS = ["aggregated", "longitudinal"]
ALL_MODES = ["icd_only", "icd_ukb_early", "icd_ukb_late"]


def _resolve_csv(data_root: str, sub_dir: str, disease: str) -> str:
    p = os.path.join(data_root, sub_dir, f"dataset_{disease}.csv")
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="包含 datasets_agg / datasets_agg_no_ukb / "
                         "datasets_longitudinal / datasets_longitudinal_no_ukb 的根目录")
    ap.add_argument("--diseases", nargs="+", default=["all"],
                    help="疾病名列表；'all' 表示跑全部 7 个疾病")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reset-out", action="store_true",
                    help="先清空 --out 文件再写入（多 disease/format 累加场景下用一次即可）")
    ap.add_argument("--formats", nargs="+", default=ALL_FORMATS, choices=ALL_FORMATS,
                    help="数据格式列表，默认 aggregated 和 longitudinal 都跑")
    ap.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES,
                    help="input 配置列表，默认三种都跑")
    args = ap.parse_args()

    diseases = ALL_DISEASES if args.diseases == ["all"] else args.diseases
    here = os.path.dirname(os.path.abspath(__file__))
    runner = os.path.join(here, "ex_medical.py")
    out_path = os.path.abspath(args.out)
    data_root = os.path.abspath(args.data_root)

    if args.reset_out and os.path.exists(out_path):
        os.remove(out_path)

    print(f"data_root: {data_root}")
    print(f"diseases:  {diseases}")
    print(f"formats:   {args.formats}")
    print(f"modes:     {args.modes}")
    print(f"out:       {out_path}")

    for disease in diseases:
        for fmt in args.formats:
            no_ukb_csv = _resolve_csv(data_root, SUBDIR_MAP[(fmt, False)], disease)
            with_ukb_csv = _resolve_csv(data_root, SUBDIR_MAP[(fmt, True)], disease)

            for mode in args.modes:
                csv = no_ukb_csv if mode == "icd_only" else with_ukb_csv
                cmd = [
                    sys.executable, runner,
                    "--csv", csv,
                    "--mode", mode,
                    "--format", fmt,
                    "--disease", disease,
                    "--epochs", str(args.epochs),
                    "--patience", str(args.patience),
                    "--seed", str(args.seed),
                    "--out", out_path,
                ]
                print("\n>> " + " ".join(cmd))
                subprocess.run(cmd, check=True)

    print(f"\nAll runs finished. Results: {out_path}")


if __name__ == "__main__":
    main()
