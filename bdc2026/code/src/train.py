# -*- coding: utf-8 -*-
"""
WX ALPHA 训练脚本

策略: A7 (0.5*规则反转 + 0.5*5基树堆叠) + M14避雷 + 冠军高换手子池
流程:
  1. 从 train.csv 构建因子面板
  2. Walk-forward OOS 评估 B0/M0/A7/A7+M14
  3. 保存评估指标到 model/wx_a7_m14/
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 确保能 import 同目录模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

import alpha_features as F
import alpha_pipeline as P
import alpha_scorers as S
import config as cfg


def _attach_m14(panel: pd.DataFrame, base_preds: pd.DataFrame) -> pd.DataFrame:
    """为 base_preds 附加 M14 安全分 (OOS)。"""
    m14 = P.predict_oos(panel, S.m14_bottom20_score, cfg.STD_FACTORS, cfg.LABEL_COL)
    if m14.empty:
        base_preds["m14_safe"] = 0.5
        return base_preds
    key = m14[["code", "date", "score"]].rename(columns={"score": "m14_safe"})
    return base_preds.merge(key, on=["code", "date"], how="left")


def _attach_label_rank(preds: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return preds
    if "label_rank" in preds.columns:
        return preds
    key = panel[["code", "date", "label_rank"]]
    return preds.merge(key, on=["code", "date"], how="left")


def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    train_csv = os.path.join(cfg.DATA_PATH, "train.csv")
    test_csv = os.path.join(cfg.DATA_PATH, "test.csv")

    print("=" * 60)
    print(f"WX ALPHA 训练 | 策略: {cfg.STRATEGY}")
    print(f"数据: {train_csv}")
    print("=" * 60)

    print("\n[1/4] 构建因子面板...")
    panel = F.build_panel(train_csv)
    print(f"  面板: {len(panel)} 行, {panel['code'].nunique()} 票, "
          f"{panel['date'].min().date()} ~ {panel['date'].max().date()}")

    results = {}

    print("\n[2/4] OOS 评估 B0 (规则反转)...")
    b0_preds = P.predict_oos(panel, S.rule_score, cfg.REV_FACTORS, cfg.LABEL_COL)
    if b0_preds.empty:
        print("  警告: B0 OOS 为空, 使用末窗全样本评估")
        dates = sorted(panel["date"].unique())
        tr_dates = dates[:-cfg.WF_VALID_DAYS]
        va_dates = dates[-cfg.WF_VALID_DAYS:]
        tr, va = panel[panel["date"].isin(tr_dates)], panel[panel["date"].isin(va_dates)].copy()
        sc = S.rule_score(tr, va, cfg.REV_FACTORS, cfg.LABEL_COL)
        extra = panel[["code", "date", "turn_5_std"]].drop_duplicates()
        b0_preds = va.merge(extra, on=["code", "date"], how="left")
        b0_preds = b0_preds.assign(score=sc)[["code", "date", "score", "label", "label_rank", "turn_5_std"]]
    else:
        b0_preds = _attach_label_rank(b0_preds, panel)
    b0_pos = P.run_champion_pipeline(b0_preds, use_subpool=True, use_m14=False)
    results["B0_rule_champion"] = P.evaluate_positions(b0_pos)
    print(f"  B0 net={results['B0_rule_champion']['net_total']:.4f} "
          f"真实分位={results['B0_rule_champion']['top_real_rank_pct']:.4f}")

    print("\n[3/4] OOS 评估 A7 (混合堆叠)...")
    a7_preds = P.predict_oos(panel, S.a7_blend_score, cfg.STD_FACTORS, cfg.LABEL_COL)
    a7_preds = _attach_label_rank(a7_preds, panel)
    a7_preds = _attach_m14(panel, a7_preds)
    a7_pos = P.run_champion_pipeline(a7_preds, use_subpool=True, use_m14=False)
    results["A7_champion"] = P.evaluate_positions(a7_pos)
    print(f"  A7 net={results['A7_champion']['net_total']:.4f} "
          f"真实分位={results['A7_champion']['top_real_rank_pct']:.4f}")

    print("\n[4/4] OOS 评估 A7+M14 (避雷)...")
    a7m14_pos = P.run_champion_pipeline(a7_preds, use_subpool=True, use_m14=True)
    results["A7_M14_champion"] = P.evaluate_positions(a7m14_pos)
    print(f"  A7+M14 net={results['A7_M14_champion']['net_total']:.4f} "
          f"真实分位={results['A7_M14_champion']['top_real_rank_pct']:.4f}")

    # 赛制测试集参考分 (用全量训练后最新日选股, 在 test.csv 上评分)
    print("\n[参考] 赛制测试集评分 (score_self 口径)...")
    latest = panel["date"].max()
    tr = panel[panel["date"] < latest - pd.Timedelta(days=cfg.WF_VALID_DAYS)]
    va = panel[panel["date"] == latest].copy()
    if len(tr) >= cfg.MIN_TRAIN_ROWS and not va.empty:
        sc = S.a7_blend_score(tr, va, cfg.STD_FACTORS, cfg.LABEL_COL)
        m14 = S.m14_bottom20_score(tr, va, cfg.STD_FACTORS, cfg.LABEL_COL)
        va = va.loc[:, ~va.columns.duplicated()].copy()
        va["score"] = sc
        va["m14_safe"] = m14
        va = P.filter_subpool(va, frac=cfg.SUBPOOL_FRAC)
        va = P.apply_m14_gate(va, va["m14_safe"].to_numpy(), cfg.M14_GATE_QUANTILE)
        top = va.nlargest(cfg.TOP_K, "score")
        hold = pd.DataFrame({
            "date": [latest] * len(top),
            "code": top["code"].values,
            "score": top["score"].values,
            "label": [np.nan] * len(top),
            "weight": [1.0 / cfg.TOP_K] * len(top),
        })
        if os.path.exists(test_csv):
            comp_score = P.competition_score(hold, test_csv)
            results["competition_test_score"] = comp_score
            print(f"  赛制测试得分: {comp_score:.6f}")

    # 基线对比 (模板 Transformer 最佳分)
    results["baseline_transformer_final_score"] = 0.037838
    results["strategy"] = cfg.STRATEGY
    results["timestamp"] = datetime.now().isoformat()
    results["a7_beta"] = cfg.A7_BETA
    results["m14_gate_quantile"] = cfg.M14_GATE_QUANTILE

    metrics_path = Path(cfg.OUTPUT_DIR) / "train_metrics.json"
    P.save_metrics(results, metrics_path)

    with open(Path(cfg.OUTPUT_DIR) / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.config, f, indent=2, ensure_ascii=False)

    # final_score.txt 供赛制目录结构兼容
    best = results["A7_M14_champion"]
    with open(Path(cfg.OUTPUT_DIR) / "final_score.txt", "w", encoding="utf-8") as f:
        f.write(f"Strategy: {cfg.STRATEGY}\n")
        f.write(f"A7+M14 net_total: {best['net_total']:.6f}\n")
        f.write(f"A7+M14 top_real_rank_pct: {best['top_real_rank_pct']:.6f}\n")
        if "competition_test_score" in results:
            f.write(f"Competition test score: {results['competition_test_score']:.6f}\n")

    print("\n" + "=" * 60)
    print(f"训练完成! 指标已保存: {metrics_path}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    main()
