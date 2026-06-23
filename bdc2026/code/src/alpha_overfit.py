# -*- coding: utf-8 -*-
"""抗过拟合套件 — PBO / DSR (loop 模式3 对抗验证)"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import norm


def pbo_cscv(net_df: pd.DataFrame, n_blocks: int = 8) -> dict:
    """CSCV PBO: IS 最优臂 OOS 掉到中位以下的概率。"""
    R = net_df.dropna(how="any")
    if R.shape[1] < 2 or R.shape[0] < n_blocks:
        return {"pbo": float("nan"), "n_combos": 0, "pass": False}
    T = R.shape[0]
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    blocks = [R.iloc[edges[i]: edges[i + 1]] for i in range(n_blocks)]

    def sharpe(df):
        mu = df.mean()
        sd = df.std(ddof=1).replace(0, np.nan)
        return mu / sd

    half = n_blocks // 2
    logits = []
    for combo in itertools.combinations(range(n_blocks), half):
        IS = pd.concat([blocks[i] for i in combo])
        oos_idx = [i for i in range(n_blocks) if i not in combo]
        OOS = pd.concat([blocks[i] for i in oos_idx])
        s_is, s_oos = sharpe(IS), sharpe(OOS)
        best = s_is.idxmax()
        r = float(s_oos.rank(pct=True)[best])
        r = min(max(r, 1e-6), 1 - 1e-6)
        logits.append(np.log(r / (1 - r)))
    pbo = float((np.array(logits) <= 0).mean())
    return {"pbo": pbo, "n_combos": len(logits), "pass": pbo < 0.5}


def deflated_sharpe(net: pd.Series, n_trials: int, sr_variance: float) -> dict:
    r = pd.Series(net).dropna().to_numpy()
    n = len(r)
    if n < 10:
        return {"sr": float("nan"), "dsr_p": float("nan"), "pass": False}
    mu, sd = r.mean(), r.std(ddof=1)
    if sd == 0:
        return {"sr": float("nan"), "dsr_p": float("nan"), "pass": False}
    sr = mu / sd
    g3 = ((r - mu) ** 3).mean() / sd ** 3
    g4 = ((r - mu) ** 4).mean() / sd ** 4
    e = 0.5772156649
    sr0 = np.sqrt(max(sr_variance, 1e-12)) * (
        (1 - e) * norm.ppf(1 - 1.0 / max(n_trials, 2))
        + e * norm.ppf(1 - 1.0 / (max(n_trials, 2) * np.e))
    )
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2, 1e-9))
    z = (sr - sr0) * np.sqrt(n - 1) / denom
    dsr_p = float(norm.cdf(z))
    return {"sr": float(sr), "sr0_threshold": float(sr0), "dsr_p": dsr_p, "pass": dsr_p >= 0.95}


def run_overfit_suite(daily_net: pd.Series, n_trials: int = 6) -> dict:
    """对单臂逐日收益做 DSR; 多臂对比时外部拼 net_df 调 pbo。"""
    sr_var = 0.01
    dsr = deflated_sharpe(daily_net, n_trials=n_trials, sr_variance=sr_var)
    return {"dsr": dsr, "n_days": int(daily_net.dropna().shape[0])}
