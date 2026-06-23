# -*- coding: utf-8 -*-
"""增强因子 v2: 在基础17因子上扩展更多alpha信号"""

from __future__ import annotations

import numpy as np
import pandas as pd

import alpha_features as F

EXTRA_FACTORS = [
    "gap_5", "smart_money", "vwap_dev_5",
    "price_pos_20", "price_pos_60",
    "ret_skew_20", "up_down_ratio_10",
    "overnight_ret_5", "intra_strength_5",
    "vol_ratio_5_20", "turn_accel",
    "rev_5_adj", "rev_10_adj", "mom_vol_ratio",
]

ALL_STD_FACTORS = F.STD_FACTORS + [f"{f}_std" for f in EXTRA_FACTORS]


def compute_extra_factors(daily: pd.DataFrame) -> pd.DataFrame:
    """计算额外alpha因子。"""
    df = daily.copy()
    g = df.groupby("code", sort=False)

    # Gap (overnight return): open / prev_close - 1
    prev_close = g["close"].shift(1)
    df["gap"] = df["open"] / prev_close.replace(0, np.nan) - 1.0
    df["gap_5"] = df.groupby("code")["gap"].transform(
        lambda s: s.rolling(5, min_periods=3).mean())

    # Smart money: (close - low) / (high - low) * volume
    intra_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["_sm"] = (df["close"] - df["low"]) / intra_range * df["volume"]
    df["smart_money"] = df.groupby("code")["_sm"].transform(
        lambda s: s.rolling(5, min_periods=3).mean())

    # VWAP deviation: close / vwap_proxy - 1
    df["_vwap"] = df["amount"] / df["volume"].replace(0, np.nan)
    df["vwap_dev_5"] = df.groupby("code").apply(
        lambda x: (x["close"] / x["_vwap"].replace(0, np.nan) - 1).rolling(5, min_periods=3).mean(),
        include_groups=False).reset_index(level=0, drop=True)

    # Price position in N-day range
    roll_high_20 = g["high"].transform(lambda s: s.rolling(20, min_periods=10).max())
    roll_low_20 = g["low"].transform(lambda s: s.rolling(20, min_periods=10).min())
    df["price_pos_20"] = (df["close"] - roll_low_20) / (roll_high_20 - roll_low_20).replace(0, np.nan)

    roll_high_60 = g["high"].transform(lambda s: s.rolling(60, min_periods=30).max())
    roll_low_60 = g["low"].transform(lambda s: s.rolling(60, min_periods=30).min())
    df["price_pos_60"] = (df["close"] - roll_low_60) / (roll_high_60 - roll_low_60).replace(0, np.nan)

    # Return skewness (20-day)
    df["_ret"] = g["close"].pct_change()
    df["ret_skew_20"] = df.groupby("code")["_ret"].transform(
        lambda s: s.rolling(20, min_periods=10).skew())

    # Up/down ratio (10-day)
    df["_up"] = (df["_ret"] > 0).astype(float)
    df["_dn"] = (df["_ret"] < 0).astype(float)
    up_sum = df.groupby("code")["_up"].transform(lambda s: s.rolling(10, min_periods=5).sum())
    dn_sum = df.groupby("code")["_dn"].transform(lambda s: s.rolling(10, min_periods=5).sum())
    df["up_down_ratio_10"] = up_sum / dn_sum.replace(0, np.nan)

    # Overnight return mean (5-day)
    df["overnight_ret_5"] = df["gap_5"]  # same as gap_5

    # Intraday strength: (close - open) / (high - low)
    df["_intra"] = (df["close"] - df["open"]) / intra_range
    df["intra_strength_5"] = df.groupby("code")["_intra"].transform(
        lambda s: s.rolling(5, min_periods=3).mean())

    # Vol ratio 5/20
    vol5 = df.groupby("code")["_ret"].transform(lambda s: s.rolling(5, min_periods=3).std())
    vol20 = df.groupby("code")["_ret"].transform(lambda s: s.rolling(20, min_periods=10).std())
    df["vol_ratio_5_20"] = vol5 / vol20.replace(0, np.nan)

    # Turnover acceleration
    turn5 = df.groupby("code")["turn"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    turn20 = df.groupby("code")["turn"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["turn_accel"] = turn5 / turn20.replace(0, np.nan) - 1.0

    # Volatility-adjusted reversal (better reversal signal)
    rev5 = -(df["close"] / g["close"].shift(5) - 1.0)
    vol_adj = vol20.replace(0, np.nan)
    df["rev_5_adj"] = rev5 / vol_adj
    rev10 = -(df["close"] / g["close"].shift(10) - 1.0)
    df["rev_10_adj"] = rev10 / vol_adj

    # Momentum / Volatility ratio (quality-adjusted momentum)
    mom = df["close"] / g["close"].shift(20) - 1.0
    df["mom_vol_ratio"] = mom / vol_adj

    return df


def cross_section_std_extra(panel: pd.DataFrame) -> pd.DataFrame:
    """对额外因子做截面标准化。"""
    out = panel.copy()
    for fac in EXTRA_FACTORS:
        if fac not in out.columns:
            continue
        col = f"{fac}_std"
        parts = []
        for _, g in out.groupby("date", sort=True):
            s = F._winsorize(g[fac].astype(float))
            gg = g.copy()
            gg[col] = F._zscore(s)
            parts.append(gg)
        out = pd.concat(parts, ignore_index=True)
    return out


def build_panel_v2(csv_path: str) -> pd.DataFrame:
    """构建增强面板: 17基础因子 + 14额外因子。"""
    daily = F.load_daily_csv(csv_path)
    raw = F.compute_raw_factors(daily)
    extra = compute_extra_factors(raw)
    std = F.cross_section_std(extra)
    std2 = cross_section_std_extra(std)
    panel = F.build_labels(std2)
    all_std = F.STD_FACTORS + [f"{f}_std" for f in EXTRA_FACTORS if f"{f}_std" in panel.columns]
    base_keep = ["code", "date"] + all_std + ["label", "label_rank"]
    if "turn_5_std" not in all_std:
        base_keep.append("turn_5_std")
    keep = list(dict.fromkeys(c for c in base_keep if c in panel.columns))
    panel = panel[keep].dropna(subset=["label"])
    return panel.sort_values(["date", "code"]).reset_index(drop=True)
