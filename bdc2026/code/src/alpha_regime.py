# -*- coding: utf-8 -*-
"""20+ 常见择时策略 — 基于市场代理指数 (train.csv 等权合成)"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

import alpha_features as F


@dataclass
class TimingSpec:
    code: str
    name: str
    fn: Callable[[pd.Series], dict]


def build_market_index(csv_path: str) -> pd.DataFrame:
    """从全市场日线构建代理指数 (等权收盘 + 衍生指标)。"""
    daily = F.load_daily_csv(csv_path)
    idx = daily.groupby("date").agg(
        close=("close", "mean"),
        volume=("volume", "sum"),
        ret=("close", lambda s: s.pct_change().mean()),
    ).sort_index()
    idx["ret"] = idx["close"].pct_change()
    for w in (5, 10, 20, 30, 60, 120, 150, 200):
        idx[f"ma{w}"] = idx["close"].rolling(w, min_periods=max(5, w // 3)).mean()
    # RSI14
    delta = idx["close"].diff()
    up = delta.clip(lower=0).rolling(14, min_periods=7).mean()
    dn = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()
    idx["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-12))
    # MACD
    ema12 = idx["close"].ewm(span=12, adjust=False).mean()
    ema26 = idx["close"].ewm(span=26, adjust=False).mean()
    idx["macd"] = ema12 - ema26
    idx["macd_sig"] = idx["macd"].ewm(span=9, adjust=False).mean()
    # 波动率
    idx["vol20"] = idx["ret"].rolling(20, min_periods=10).std()
    idx["vol60"] = idx["ret"].rolling(60, min_periods=20).std()
    # 回撤
    idx["peak"] = idx["close"].cummax()
    idx["dd"] = idx["close"] / idx["peak"] - 1.0
    # 252日高点距离
    idx["high252"] = idx["close"].rolling(252, min_periods=60).max()
    idx["dist_high252"] = idx["close"] / idx["high252"] - 1.0
    return idx


def _state_map(idx: pd.DataFrame, mask_strong, mask_weak, weak_scale: float = 0.25) -> dict:
    """三态 -> 仓位系数。"""
    out = {}
    for d in idx.index:
        if bool(mask_strong.loc[d]) if d in mask_strong.index else False:
            out[d] = 1.0
        elif bool(mask_weak.loc[d]) if d in mask_weak.index else False:
            out[d] = weak_scale
        else:
            out[d] = 1.0
    return out


def _binary_map(idx: pd.DataFrame, mask_on: pd.Series, off_scale: float = 0.0) -> dict:
    return {d: (1.0 if bool(mask_on.loc[d]) else off_scale) for d in idx.index if d in mask_on.index}


def _continuous_map(idx: pd.DataFrame, signal: pd.Series, lo: float, hi: float) -> dict:
    """连续信号线性映射到 [lo, hi]。"""
    s = signal.reindex(idx.index).fillna(0.0)
    smin, smax = float(s.min()), float(s.max())
    if smax - smin < 1e-12:
        return {d: hi for d in idx.index}
    out = {}
    for d in idx.index:
        z = (float(s.loc[d]) - smin) / (smax - smin)
        out[d] = lo + z * (hi - lo)
    return out


def build_breadth(daily_path: str, window: int = 20) -> pd.Series:
    """上涨广度: 收盘>MA(window) 的股票占比。"""
    daily = F.load_daily_csv(daily_path)
    daily = daily.sort_values(["code", "date"])
    daily["ma"] = daily.groupby("code")["close"].transform(
        lambda s: s.rolling(window, min_periods=max(3, window // 2)).mean())
    daily["above"] = (daily["close"] > daily["ma"]).astype(float)
    return daily.groupby("date")["above"].mean()


def all_timing_specs(daily_path: str) -> list[TimingSpec]:
    """注册 24 种择时策略 (含冠军基准)。"""
    idx = build_market_index(daily_path)
    breadth20 = build_breadth(daily_path, 20)
    breadth60 = build_breadth(daily_path, 60)
    specs: list[TimingSpec] = []

    # T00 无择时
    specs.append(TimingSpec("T00", "无择时(满仓1.0)", lambda i: {d: 1.0 for d in i.index}))

    # T01 冠军 MA60/150
    def t01(i):
        strong = (i["close"] > i["ma60"]) & (i["close"] > i["ma150"])
        weak = (i["close"] < i["ma60"]) & (i["close"] < i["ma150"])
        return _state_map(i, strong, weak, 0.25)
    specs.append(TimingSpec("T01", "冠军MA60/150(弱0.25)", t01))

    # T02 MA20/60
    def t02(i):
        strong = (i["close"] > i["ma20"]) & (i["close"] > i["ma60"])
        weak = (i["close"] < i["ma20"]) & (i["close"] < i["ma60"])
        return _state_map(i, strong, weak, 0.25)
    specs.append(TimingSpec("T02", "MA20/60三态", t02))

    # T03 MA50/200
    def t03(i):
        strong = (i["close"] > i["ma60"]) & (i["close"] > i["ma200"])
        weak = (i["close"] < i["ma60"]) & (i["close"] < i["ma200"])
        return _state_map(i, strong, weak, 0.25)
    specs.append(TimingSpec("T03", "MA50/200三态", t03))

    # T04 MA10/30 短周期
    def t04(i):
        strong = (i["close"] > i["ma10"]) & (i["close"] > i["ma30"])
        weak = (i["close"] < i["ma10"]) & (i["close"] < i["ma30"])
        return _state_map(i, strong, weak, 0.5)
    specs.append(TimingSpec("T04", "MA10/30短周期", t04))

    # T05 价格>MA60 二元
    specs.append(TimingSpec("T05", "价格>MA60二元", lambda i: _binary_map(i, i["close"] > i["ma60"], 0.0)))

    # T06 价格>MA120
    specs.append(TimingSpec("T06", "价格>MA120二元", lambda i: _binary_map(i, i["close"] > i["ma120"], 0.0)))

    # T07 MA20斜率为正
    ma20_slope = idx["ma20"].diff(5)
    specs.append(TimingSpec("T07", "MA20斜率>0", lambda i: _binary_map(i, ma20_slope > 0, 0.3)))

    # T08 RSI择时
    def t08(i):
        out = {}
        for d in i.index:
            r = float(i.loc[d, "rsi14"]) if not pd.isna(i.loc[d, "rsi14"]) else 50
            if r >= 50:
                out[d] = 1.0
            elif r <= 30:
                out[d] = 0.25
            else:
                out[d] = 0.6
        return out
    specs.append(TimingSpec("T08", "RSI14三档", t08))

    # T09 低波满仓高波减仓
    vol_med = idx["vol20"].median()
    def t09(i):
        out = {}
        for d in i.index:
            v = float(i.loc[d, "vol20"]) if not pd.isna(i.loc[d, "vol20"]) else vol_med
            out[d] = 1.0 if v <= vol_med else max(0.25, 1.0 - (v / (vol_med + 1e-12) - 1))
        return out
    specs.append(TimingSpec("T09", "波动率分档", t09))

    # T10 回撤控制
    def t10(i):
        out = {}
        for d in i.index:
            dd = float(i.loc[d, "dd"]) if not pd.isna(i.loc[d, "dd"]) else 0
            if dd >= -0.05:
                out[d] = 1.0
            elif dd >= -0.10:
                out[d] = 0.5
            else:
                out[d] = 0.25
        return out
    specs.append(TimingSpec("T10", "回撤阶梯", t10))

    # T11 指数20日动量
    mom20 = idx["close"].pct_change(20)
    specs.append(TimingSpec("T11", "指数20日动量>0", lambda i: _binary_map(i, mom20 > 0, 0.25)))

    # T12 广度>50%
    b20 = breadth20.reindex(idx.index).fillna(0.5)
    specs.append(TimingSpec("T12", "广度>50%", lambda i: _binary_map(i, b20 > 0.5, 0.3)))

    # T13 三重过滤 MA60+广度
    def t13(i):
        on = (i["close"] > i["ma60"]) & (b20 > 0.5)
        return _binary_map(i, on, 0.0)
    specs.append(TimingSpec("T13", "MA60+广度双确认", t13))

    # T14 高波减半
    def t14(i):
        out = {}
        for d in i.index:
            v = float(i.loc[d, "vol20"]) if not pd.isna(i.loc[d, "vol20"]) else 0
            out[d] = 0.5 if v > vol_med * 1.5 else 1.0
        return out
    specs.append(TimingSpec("T14", "极高波减半", t14))

    # T15 趋势强度连续
    trend = idx["close"] / idx["ma60"] - 1.0
    specs.append(TimingSpec("T15", "趋势强度连续", lambda i: _continuous_map(i, trend, 0.25, 1.0)))

    # T16 MACD信号
    specs.append(TimingSpec("T16", "MACD>信号线", lambda i: _binary_map(i, i["macd"] > i["macd_sig"], 0.35)))

    # T17 布林位置 (close vs ma20 ± 2*std)
    std20 = idx["close"].rolling(20, min_periods=10).std()
    upper = idx["ma20"] + 2 * std20
    lower = idx["ma20"] - 2 * std20
    def t17(i):
        out = {}
        for d in i.index:
            c, u, l = i.loc[d, "close"], upper.loc[d], lower.loc[d]
            if pd.isna(u) or pd.isna(l):
                out[d] = 1.0
            elif c > u:
                out[d] = 0.5  # 超买减仓
            elif c < l:
                out[d] = 0.75  # 超卖略加仓
            else:
                out[d] = 1.0
        return out
    specs.append(TimingSpec("T17", "布林带位置", t17))

    # T18 距252日高点
    def t18(i):
        out = {}
        for d in i.index:
            dist = float(i.loc[d, "dist_high252"]) if not pd.isna(i.loc[d, "dist_high252"]) else 0
            out[d] = 1.0 if dist >= -0.05 else (0.5 if dist >= -0.15 else 0.25)
        return out
    specs.append(TimingSpec("T18", "距年高阶梯", t18))

    # T19 冠军弱市0.5 (更温和)
    def t19(i):
        strong = (i["close"] > i["ma60"]) & (i["close"] > i["ma150"])
        weak = (i["close"] < i["ma60"]) & (i["close"] < i["ma150"])
        return _state_map(i, strong, weak, 0.5)
    specs.append(TimingSpec("T19", "冠军MA60/150(弱0.5)", t19))

    # T20 波动率目标 (target vol 2%)
    target_vol = 0.02
    def t20(i):
        out = {}
        for d in i.index:
            v = float(i.loc[d, "vol20"]) if not pd.isna(i.loc[d, "vol20"]) else target_vol
            out[d] = min(1.0, max(0.25, target_vol / (v + 1e-12)))
        return out
    specs.append(TimingSpec("T20", "波动率目标2%", t20))

    # T21 季节性 (弱4-5月)
    def t21(i):
        out = {}
        for d in i.index:
            m = d.month
            out[d] = 0.5 if m in (4, 5) else 1.0
        return out
    specs.append(TimingSpec("T21", "季节性4-5月减仓", t21))

    # T22 广度60
    b60 = breadth60.reindex(idx.index).fillna(0.5)
    specs.append(TimingSpec("T22", "广度60>55%", lambda i: _binary_map(i, b60 > 0.55, 0.3)))

    # T23 双均线+MACD
    def t23(i):
        on = (i["close"] > i["ma60"]) & (i["macd"] > i["macd_sig"])
        return _binary_map(i, on, 0.25)
    specs.append(TimingSpec("T23", "MA60+MACD双确认", t23))

    # T24 组合: 冠军择时 × 波动率目标
    def t24(i):
        base = t01(i)
        vol = t20(i)
        return {d: min(base.get(d, 1.0), vol.get(d, 1.0)) for d in i.index}
    specs.append(TimingSpec("T24", "冠军×波动率目标", t24))

    # T25 条件择时: 指数>=MA150 满仓; 跌破MA150 才启用 T01 弱仓逻辑
    def t25(i):
        out = {}
        for d in i.index:
            c = i.loc[d, "close"]
            ma60 = i.loc[d, "ma60"]
            ma150 = i.loc[d, "ma150"]
            if pd.isna(ma150) or pd.isna(ma60):
                out[d] = 1.0
            elif c >= ma150:
                out[d] = 1.0
            elif c < ma60 and c < ma150:
                out[d] = 0.25
            else:
                out[d] = 1.0
        return out
    specs.append(TimingSpec("T25", "条件择时(破MA150才弱仓)", t25))

    return specs
