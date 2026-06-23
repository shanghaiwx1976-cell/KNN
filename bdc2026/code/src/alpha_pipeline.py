# -*- coding: utf-8 -*-
"""冠军下游链 + Walk-forward OOS + 赛制评分"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import alpha_scorers as S
import alpha_portfolio as PF
import config as cfg

try:
    import alpha_overlay as OV
except ImportError:
    OV = None


def walk_forward_splits(dates: list, train_days: int, valid_days: int,
                        step_days: int, embargo_days: int) -> list[tuple]:
    """时序 walk-forward 切分 (purge + embargo)。"""
    dates = sorted(pd.to_datetime(dates))
    n = len(dates)
    splits = []
    start = 0
    while True:
        tr_end = start + train_days
        va_start = tr_end + embargo_days
        va_end = va_start + valid_days
        if va_end > n:
            break
        tr_dates = dates[start:tr_end]
        va_dates = dates[va_start:va_end]
        if tr_dates and va_dates:
            splits.append((tr_dates, va_dates))
        start += step_days
    return splits


def filter_subpool(df: pd.DataFrame, col: str = "turn_5_std", frac: float = 1.0 / 3) -> pd.DataFrame:
    """冠军高换手子池: 每日保留 turn_5_std 上 frac 分位。"""
    df = df.loc[:, ~df.columns.duplicated()].copy()
    parts = []
    for _, g in df.groupby("date"):
        q = g[col].quantile(1 - frac)
        parts.append(g[g[col] >= q])
    return pd.concat(parts, ignore_index=True)


def apply_m14_gate(df: pd.DataFrame, m14_scores: np.ndarray, quantile: float) -> pd.DataFrame:
    """M14 避雷门: 剔除 m14 分数低于截面 quantile 的票。"""
    out = df.copy()
    out["m14_safe"] = m14_scores
    parts = []
    for _, g in out.groupby("date"):
        thr = g["m14_safe"].quantile(quantile)
        parts.append(g[g["m14_safe"] >= thr])
    return pd.concat(parts, ignore_index=True)


def select_topk(df: pd.DataFrame, k: int = 5, score_col: str = "score") -> pd.DataFrame:
    """每日 Top-K 等权。"""
    rows = []
    for d, g in df.groupby("date"):
        top = g.nlargest(k, score_col)
        for _, r in top.iterrows():
            rows.append({
                "date": d,
                "code": r["code"],
                "score": r[score_col],
                "label": r.get("label", np.nan),
                "weight": 1.0 / k,
            })
    return pd.DataFrame(rows)


def predict_oos(panel: pd.DataFrame, scorer, feats: list[str], label: str) -> pd.DataFrame:
    """Walk-forward OOS 打分。"""
    dates = sorted(panel["date"].unique())
    splits = walk_forward_splits(
        dates, cfg.WF_TRAIN_DAYS, cfg.WF_VALID_DAYS,
        cfg.WF_STEP_DAYS, cfg.EMBARGO_DAYS)
    preds = []
    for tr_dates, va_dates in splits:
        tr = panel[panel["date"].isin(tr_dates)]
        va = panel[panel["date"].isin(va_dates)].copy()
        if len(tr) < cfg.MIN_TRAIN_ROWS or va.empty:
            continue
        sc = scorer(tr, va, feats, label)
        va = va.assign(score=sc)
        preds.append(va[["code", "date", "score", "label", "label_rank", "turn_5_std"]])
    if not preds:
        return pd.DataFrame()
    return pd.concat(preds, ignore_index=True)


def drawdown_protection(
    positions: pd.DataFrame,
    dd_threshold: float = -0.10,
    dd_reduce: float = 0.30,
    lookback: int = 20,
) -> pd.DataFrame:
    """回撤保护: 基于前一日回撤信号减仓, 无未来信息泄露。

    逻辑: 每日组合收益累积 -> 计算相对滚动高点的回撤 -> 若前一日回撤超阈值则当日减仓。
    """
    if positions.empty:
        return positions
    daily_ret = positions.groupby("date").apply(
        lambda g: float((g["label"] * g["weight"]).sum())
    ).sort_index()
    if len(daily_ret) < lookback:
        return positions

    cum_ret = (1 + daily_ret).cumprod()
    rolling_peak = cum_ret.rolling(lookback, min_periods=5).max()
    rolling_dd = cum_ret / rolling_peak - 1.0

    pos = positions.copy()
    dates_sorted = sorted(pos["date"].unique())
    for i, d in enumerate(dates_sorted):
        if i < 1:
            continue
        prev_d = dates_sorted[i - 1]
        if prev_d not in rolling_dd.index:
            continue
        dd = float(rolling_dd.loc[prev_d]) if not pd.isna(rolling_dd.loc[prev_d]) else 0
        if dd < dd_threshold:
            mask = pos["date"] == d
            pos.loc[mask, "weight"] = pos.loc[mask, "weight"] * dd_reduce
    return pos


def run_champion_pipeline(
    preds: pd.DataFrame,
    use_subpool: bool = True,
    use_m14: bool = False,
    use_buffer: bool = True,
    regime_map: dict | None = None,
    use_vol_target: bool | None = None,
    use_dd_protect: bool | None = None,
) -> pd.DataFrame:
    """冠军下游: 子池 -> [M14门] -> 缓冲TopK -> [regime] -> [vol-target] -> [回撤保护]。"""
    df = preds.loc[:, ~preds.columns.duplicated()].copy()
    if use_subpool:
        df = filter_subpool(df, frac=cfg.SUBPOOL_FRAC)
    if use_m14 and "m14_safe" in df.columns:
        df = apply_m14_gate(df, df["m14_safe"].to_numpy(), cfg.M14_GATE_QUANTILE)
    if use_buffer:
        pos = PF.select_topk_buffered(df, k=cfg.TOP_K, keep_band=4 * cfg.TOP_K, replace_band=6 * cfg.TOP_K)
        key = df[["date", "code", "label", "label_rank", "score"]].drop_duplicates(["date", "code"])
        pos = pos.merge(key, on=["date", "code"], how="left")
    else:
        pos = select_topk(df, k=cfg.TOP_K)
    if regime_map:
        pos = PF.apply_regime_sizer(pos, regime_map)
    vt = cfg.USE_VOL_TARGET if use_vol_target is None else use_vol_target
    if vt and OV is not None:
        pos = OV.apply_vol_target(pos)
    dd = cfg.USE_DD_PROTECT if use_dd_protect is None else use_dd_protect
    if dd:
        pos = drawdown_protection(pos, cfg.DD_THRESHOLD, cfg.DD_REDUCE, cfg.DD_LOOKBACK)
    return pos


def daily_returns_from_positions(positions: pd.DataFrame) -> pd.Series:
    if positions.empty:
        return pd.Series(dtype=float)
    return positions.groupby("date").apply(
        lambda g: float((g["label"] * g["weight"]).sum())
    ).sort_index()


def evaluate_positions(positions: pd.DataFrame, attach_ranks: pd.DataFrame | None = None) -> dict:
    """评估持仓: 加权收益、IC、真实分位、最大回撤。"""
    if positions.empty:
        return {"net_total": 0.0, "precision_at5": 0.0, "top_real_rank_pct": 0.5, "n_days": 0, "max_drawdown": 0.0}

    pos = positions.copy()
    if attach_ranks is not None and "label_rank" not in pos.columns:
        pos = pos.merge(
            attach_ranks[["date", "code", "label_rank"]].drop_duplicates(),
            on=["date", "code"], how="left",
        )

    pnl = (pos["label"] * pos["weight"]).groupby(pos["date"]).sum()
    net_curve = (1 + pnl.fillna(0)).cumprod()
    peak = net_curve.cummax()
    mdd = float((net_curve / peak - 1).min()) if len(net_curve) else 0.0
    net = float(pnl.sum())

    prec, ranks = [], []
    for _, g in pos.groupby("date"):
        if "label_rank" in g.columns:
            prec.append(float((g["label_rank"] >= 0.8).mean()))
            ranks.extend(g["label_rank"].dropna().tolist())

    invested = pos.groupby("date")["weight"].sum()
    return {
        "net_total": net,
        "precision_at5": float(np.mean(prec)) if prec else 0.0,
        "top_real_rank_pct": float(np.mean(ranks)) if ranks else 0.5,
        "n_days": int(pos["date"].nunique()),
        "mean_daily_pnl": float(pnl.mean()) if len(pnl) else 0.0,
        "max_drawdown": mdd,
        "mean_invested": float(invested.mean()) if len(invested) else 1.0,
        "frac_win_days": float((pnl > 0).mean()) if len(pnl) else 0.0,
    }


def competition_score(positions: pd.DataFrame, test_csv: str) -> float:
    """按赛制 score_self.py 口径计算测试集得分。"""
    test = pd.read_csv(test_csv)
    test["股票代码"] = test["股票代码"].astype(str).str.zfill(6)
    test["日期"] = pd.to_datetime(test["日期"])

    codes = positions["code"].astype(str).str.zfill(6).tolist()
    weights = positions["weight"].tolist()
    out = pd.DataFrame({"股票代码": codes, "权重": weights})

    td = test[test["股票代码"].isin(out["股票代码"])]
    td = td.groupby("股票代码").tail(5)

    def calc_ret(g):
        s, e = g.iloc[0], g.iloc[-1]
        return (e["开盘"] - s["开盘"]) / s["开盘"]

    rets = td.groupby("股票代码").apply(calc_ret).reset_index(name="收益率")
    merged = rets.merge(out, on="股票代码")
    return float((merged["收益率"] * merged["权重"]).sum())


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
