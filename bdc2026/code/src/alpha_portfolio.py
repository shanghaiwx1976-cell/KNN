# -*- coding: utf-8 -*-
"""缓冲选股 — 冠军迟滞带 Top5 (进20/出30)"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg


def select_topk_buffered(
    scores: pd.DataFrame,
    k: int | None = None,
    keep_band: int = 20,
    replace_band: int = 30,
    date_col: str = "date",
    code_col: str = "code",
    score_col: str = "score",
) -> pd.DataFrame:
    """跨日有状态 Top-K: 持仓排名<=replace_band保留, 新仓需排名<=keep_band。"""
    k = cfg.TOP_K if k is None else k
    out_rows = []
    held: list[str] = []
    for d in sorted(scores[date_col].unique()):
        g = scores[scores[date_col] == d].sort_values(score_col, ascending=False).reset_index(drop=True)
        rank = {code: i + 1 for i, code in enumerate(g[code_col].tolist())}
        stayers = [c for c in held if c in rank and rank[c] <= replace_band]
        need = k - len(stayers)
        cands = [c for c in g[code_col].tolist() if c not in stayers and rank.get(c, 999) <= keep_band]
        port = stayers + cands[: max(0, need)]
        if not port:
            held = []
            continue
        w = 1.0 / len(port)
        for rank_i, code in enumerate(port, 1):
            out_rows.append({date_col: d, code_col: code, "weight": w, "rank_in_port": rank_i})
        held = port
    return pd.DataFrame(out_rows)


def apply_regime_sizer(
    positions: pd.DataFrame,
    regime_map: dict,
    ladder: dict | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """按择时信号缩放权重 (弱市留底/空仓)。"""
    ladder = ladder or {"strong": 1.0, "neutral": 1.0, "weak": 0.25}
    p = positions.copy()
    factors = []
    for d in p[date_col]:
        key = pd.Timestamp(d)
        state = regime_map.get(key, regime_map.get(d, "neutral"))
        if isinstance(state, (int, float)):
            factors.append(float(state))
        else:
            factors.append(float(ladder.get(str(state), ladder.get("neutral", 1.0))))
    p["weight"] = p["weight"].to_numpy(dtype=float) * np.asarray(factors, dtype=float)
    return p
