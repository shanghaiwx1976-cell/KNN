# -*- coding: utf-8 -*-
"""
WX ALPHA 预测 — A7 + 子池 + 缓冲 + regime择时
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import alpha_features as F
import alpha_pipeline as P
import alpha_portfolio as PF
import alpha_regime as R
import alpha_scorers as S
import config as cfg


def _get_regime_map(train_csv: str) -> dict:
    specs = {s.code: s for s in R.all_timing_specs(train_csv)}
    code = cfg.DEFAULT_TIMING
    if code not in specs:
        code = "T25"
    idx = R.build_market_index(train_csv)
    return specs[code].fn(idx)


def main():
    train_csv = os.path.join(cfg.DATA_PATH, "train.csv")
    output_path = os.path.join("./output/", "result.csv")
    os.makedirs("./output", exist_ok=True)

    print(f"WX ALPHA 预测 | {cfg.STRATEGY} | 择时={cfg.DEFAULT_TIMING}")

    daily = F.load_daily_csv(train_csv)
    raw = F.compute_raw_factors(daily)
    std = F.cross_section_std(raw)
    panel = F.build_labels(std)

    latest = panel["date"].max()
    cutoff = latest - pd.Timedelta(days=cfg.WF_VALID_DAYS)
    tr = panel[panel["date"] <= cutoff].dropna(subset=[cfg.LABEL_COL])
    va = panel[panel["date"] == latest].copy()

    if len(tr) < cfg.MIN_TRAIN_ROWS:
        raise ValueError(f"训练样本不足: {len(tr)}")

    a7_score = S.a7_blend_score(tr, va, cfg.STD_FACTORS, cfg.LABEL_COL)
    va = va.loc[:, ~va.columns.duplicated()].copy()
    va["score"] = a7_score
    va = P.filter_subpool(va, frac=cfg.SUBPOOL_FRAC)

    if len(va) < cfg.TOP_K:
        va = panel[panel["date"] == latest].copy()
        va["score"] = a7_score
        va = P.filter_subpool(va, frac=cfg.SUBPOOL_FRAC)

    top = va.nlargest(cfg.TOP_K, "score")
    regime_map = _get_regime_map(train_csv)
    scale = float(regime_map.get(pd.Timestamp(latest), regime_map.get(latest, 1.0)))

    if cfg.USE_VOL_TARGET:
        # 用训练窗末段组合波动近似 vol-target 缩放
        hist = panel[panel["date"] <= cutoff].dropna(subset=[cfg.LABEL_COL])
        hist_dates = sorted(hist["date"].unique())[-cfg.VOL_WINDOW * 2:]
        hsub = hist[hist["date"].isin(hist_dates)]
        if len(hsub) > cfg.MIN_TRAIN_ROWS:
            hsc = S.a7_blend_score(
                hist[hist["date"] < hist_dates[0]], hsub, cfg.STD_FACTORS, cfg.LABEL_COL)
            hsub = hsub.copy()
            hsub["score"] = hsc
            hsub = P.filter_subpool(hsub)
            hpos = P.run_champion_pipeline(
                hsub, use_subpool=False, use_buffer=True, regime_map=regime_map, use_vol_target=False)
            dret = P.daily_returns_from_positions(hpos)
            if len(dret) >= 5:
                import alpha_overlay as OV
                exp = float(OV.exposure_from_vol(dret).iloc[-1])
                scale = min(scale, exp)

    w = min(cfg.WEIGHT * scale, cfg.WEIGHT)
    if w <= 0:
        w = cfg.WEIGHT * 0.25

    codes = top["code"].astype(str).str.zfill(6).tolist()
    out = pd.DataFrame({"stock_id": codes, "weight": [w] * len(codes)})
    # 归一化使权重和 <= 1
    total = out["weight"].sum()
    if total > 1.0:
        out["weight"] = out["weight"] / total

    out.to_csv(output_path, index=False)
    print(f"预测截面: {latest.date()} | 择时仓位: {scale:.3f}")
    print(f"选股: {codes}")
    print(f"权重: {out['weight'].tolist()} (sum={out['weight'].sum():.3f})")
    print(f"结果: {output_path}")


if __name__ == "__main__":
    main()
