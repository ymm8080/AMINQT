"""parallel_feat_worker.py — legacy 双板特征构建子进程 (2026-08-21).

父进程 (DailySelectionPipeline._build_features_parallel) 把清洗帧落盘后, 按板各起
一个本模块子进程: 读板帧 → FeatureEngineV35().build → 特征帧落盘. build 是无状态
纯函数链 (dim 方法只读输入帧+config), main/dual 帧股票集不相交 → 与父进程串行
build 逐字节一致 (tests/test_parallel_feat_worker.py 验证).

用法 (父进程拼 argv):
  python -m app.pipeline1.parallel_feat_worker <in.parquet> <out.parquet> \\
      <json_inference_cols> <0|1 cs_rank> <json_float_shares_map>
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .feature_engine_v35 import FeatureEngineV35


def build_to_parquet(
    in_path: str,
    out_path: str,
    inference_cols: list[str] | None,
    cross_sectional_rank: bool,
    float_shares_map: dict | None,
) -> None:
    """读板帧 parquet → 构建特征 → 特征帧 parquet (可独立单测/手动跑)."""
    df = pd.read_parquet(in_path)
    feat = FeatureEngineV35().build(
        df,
        float_shares_map or None,
        inference_cols=inference_cols or None,
        cross_sectional_rank=cross_sectional_rank,
    )
    feat.to_parquet(out_path, index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="legacy 单板特征构建子进程")
    ap.add_argument("in_path")
    ap.add_argument("out_path")
    ap.add_argument("inference_cols", default="[]")  # JSON list[str]
    ap.add_argument("cs_rank", default="0")  # "1"=cross_sectional_rank
    ap.add_argument("float_shares", default="{}")  # JSON dict
    args = ap.parse_args(argv)
    build_to_parquet(
        args.in_path,
        args.out_path,
        json.loads(args.inference_cols),
        args.cs_rank == "1",
        json.loads(args.float_shares),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
