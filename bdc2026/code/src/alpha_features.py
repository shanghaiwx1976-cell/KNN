# -*- coding: utf-8 -*-
"""从赛制日线数据计算 ALPHA 因子 (PIT 安全, 截面 z-score)"""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_FACTORS = [
    "rev_5", "rev_10", "rev_20", "mom_20",
    "turn_5", "turn_chg", "amihud_20",
    "vol_20", "dnvol_20", "idio_vol_20",
    "amp_5", "volr_5", "lowsh_5", "upsh_5",
    "rev_turn", "rev_vol", "rev_volr",
]

STD_FACTORS = [f"{f}_std" for f in RAW_FACTORS]


def _winsorize(s: pd.Series, k: float = 3.0) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if mad < 1e-12:
        return s
    lo, hi = med - k * 1.4826 * mad, med + k * 1.4826 * mad
    return s.clip(lo, hi)


def _zscore(s: pd.Series) -> pd.Series:
    m, sd = s.mean(), s.std()
    if sd < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - m) / sd


def load_daily_csv(path: str) -> pd.DataFrame:
    """读取赛制 train.csv 并统一列名。"""
    df = pd.read_csv(path, dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])
    rename = {
        "股票代码": "code",
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turn",
    }
    out = df.rename(columns=rename)
    for c in ("open", "close", "high", "low", "volume", "amount", "turn"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.sort_values(["code", "date"]).reset_index(drop=True)
    return out


def compute_raw_factors(daily: pd.DataFrame) -> pd.DataFrame:
    """逐票计算原始因子。"""
    df = daily.copy()
    g = df.groupby("code", sort=False)

    df["ret"] = g["close"].pct_change()
    df["rev_5"] = -(df["close"] / g["close"].shift(5) - 1.0)
    df["rev_10"] = -(df["close"] / g["close"].shift(10) - 1.0)
    df["rev_20"] = -(df["close"] / g["close"].shift(20) - 1.0)
    df["mom_20"] = df["close"] / g["close"].shift(20) - 1.0

    df["turn_5"] = g["turn"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    df["turn_chg"] = df["turn"] / df["turn_5"].replace(0, np.nan) - 1.0

    ami = df["ret"].abs() / df["amount"].replace(0, np.nan) * 1e8
    df["amihud_20"] = ami.groupby(df["code"]).transform(lambda s: s.rolling(20, min_periods=10).mean())

    df["vol_20"] = g["ret"].transform(lambda s: s.rolling(20, min_periods=10).std())
    neg = df["ret"].where(df["ret"] < 0)
    df["dnvol_20"] = neg.groupby(df["code"]).transform(lambda s: s.rolling(20, min_periods=5).std())

    # idio vol proxy
    df["idio_vol_20"] = df["vol_20"]

    preclose = g["close"].shift(1)
    amp = (df["high"] - df["low"]) / preclose.replace(0, np.nan)
    df["amp_5"] = amp.groupby(df["code"]).transform(lambda s: s.rolling(5, min_periods=3).mean())
    df["volr_5"] = df["volume"] / g["volume"].transform(
        lambda s: s.rolling(5, min_periods=3).mean()).replace(0, np.nan)

    df["lowsh_5"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["upsh_5"] = (df["high"] - df["close"]) / (df["high"] - df["low"]).replace(0, np.nan)

    df["rev_turn"] = df["rev_5"] * df["turn_5"]
    df["rev_vol"] = df["rev_5"] * df["vol_20"]
    df["rev_volr"] = df["rev_5"] * df["volr_5"]

    return df


def cross_section_std(panel: pd.DataFrame) -> pd.DataFrame:
    """逐日截面 winsorize + z-score。"""
    out = panel.copy()
    for fac in RAW_FACTORS:
        col = f"{fac}_std"
        parts = []
        for _, g in out.groupby("date", sort=True):
            s = _winsorize(g[fac].astype(float))
            gg = g.copy()
            gg[col] = _zscore(s)
            parts.append(gg)
        out = pd.concat(parts, ignore_index=True)
    return out


def build_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """构建赛制标签: T+1 开盘买 -> T+5 开盘卖。"""
    df = panel.copy()
    g = df.groupby("code", sort=False)
    df["open_t1"] = g["open"].shift(-1)
    df["open_t5"] = g["open"].shift(-5)
    df["label"] = (df["open_t5"] - df["open_t1"]) / (df["open_t1"] + 1e-12)
    df["label_rank"] = df.groupby("date")["label"].rank(pct=True)
    return df


def build_panel(csv_path: str) -> pd.DataFrame:
    """完整面板: code, date, 17*_std, label, label_rank。"""
    daily = load_daily_csv(csv_path)
    raw = compute_raw_factors(daily)
    std = cross_section_std(raw)
    panel = build_labels(std)
    keep = ["code", "date"] + STD_FACTORS + ["label", "label_rank"]
    panel = panel[keep].dropna(subset=["label"])
    return panel.sort_values(["date", "code"]).reset_index(drop=True)
