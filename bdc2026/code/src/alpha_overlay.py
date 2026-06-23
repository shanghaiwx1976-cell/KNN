# -*- coding: utf-8 -*-
"""组合级风控叠加 — Devin vol-target"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg


def daily_portfolio_returns(positions: pd.DataFrame) -> pd.Series:
    """逐日组合收益 (label × weight)。"""
    if positions.empty or "label" not in positions.columns:
        return pd.Series(dtype=float)
    return positions.groupby("date").apply(
        lambda g: float((g["label"] * g["weight"]).sum())
    ).sort_index()


def exposure_from_vol(
    net: pd.Series,
    target_vol: float | None = None,
    window: int | None = None,
    cap: float = 1.0,
) -> pd.Series:
    """PIT 安全: 用 shift(1) 滚动波动率定仓位。"""
    target_vol = cfg.VOL_TARGET if target_vol is None else target_vol
    window = cfg.VOL_WINDOW if window is None else window
    vol = net.shift(1).rolling(window, min_periods=max(4, window // 3)).std(ddof=0)
    exp = (target_vol / vol).clip(upper=cap)
    return exp.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, cap)


def scale_positions(positions: pd.DataFrame, exposure: pd.Series) -> pd.DataFrame:
    out = positions.copy()
    ex = {pd.Timestamp(k): float(v) for k, v in exposure.items()}
    out["weight"] = out["weight"] * out["date"].map(lambda d: ex.get(pd.Timestamp(d), 1.0))
    return out


def apply_vol_target(positions: pd.DataFrame) -> pd.DataFrame:
    """在已有持仓上叠加 vol-target (不改选股)。"""
    if not cfg.USE_VOL_TARGET:
        return positions
    net = daily_portfolio_returns(positions)
    if len(net) < 10:
        return positions
    exp = exposure_from_vol(net)
    return scale_positions(positions, exp)
