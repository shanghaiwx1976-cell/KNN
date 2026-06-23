# -*- coding: utf-8 -*-
"""ALPHA 打分器: 规则反转 / 5基树堆叠 / M14避雷"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

import config as cfg

SEED = cfg.SEED
RANK_BUCKETS = cfg.LGB_RANK_BUCKETS
MIN_TRAIN_ROWS = cfg.MIN_TRAIN_ROWS

BASE_SPECS = [
    ("lgbreg", "label", "reg_lgb"),
    ("lgbrank", "label_rank", "rank_lgb"),
    ("histgb", "label", "reg_histgb"),
    ("rf", "label", "reg_rf"),
    ("et", "label", "reg_et"),
]


def _X(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _group_sizes(df: pd.DataFrame) -> list[int]:
    return df.groupby("date", sort=True).size().tolist()


def _xs_rank(df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    s = pd.Series(values, index=df.index)
    return df.assign(_v=s).groupby("date")["_v"].rank(pct=True).to_numpy()


def _rank_ic(train_df: pd.DataFrame, feat: str, label: str) -> float:
    ics = []
    for _, g in train_df.groupby("date"):
        if len(g) < 10:
            continue
        x = g[feat].astype(float)
        y = g[label].astype(float)
        if x.std() < 1e-12 or y.std() < 1e-12:
            continue
        ics.append(x.corr(y, method="spearman"))
    return float(np.nanmean(ics)) if ics else 0.0


def _new_model(kind: str, seed: int):
    if kind == "reg_lgb":
        p = dict(cfg.LGB_PARAMS)
        p["random_state"] = seed
        return LGBMRegressor(**p)
    if kind == "rank_lgb":
        p = dict(cfg.LGB_PARAMS)
        p["random_state"] = seed
        return LGBMRanker(objective="lambdarank", **p)
    if kind == "reg_histgb":
        return HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.05, max_iter=300,
            l2_regularization=1.0, min_samples_leaf=50, random_state=seed)
    if kind == "reg_rf":
        return RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=50,
            max_features="sqrt", n_jobs=-1, random_state=seed)
    if kind == "reg_et":
        return ExtraTreesRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=50,
            max_features="sqrt", n_jobs=-1, random_state=seed)
    raise ValueError(kind)


def rule_score(train_df, valid_df, feats, label):
    """冠军规则反转分: 训练窗 rank-IC 定符号后等权合成。"""
    tr = train_df.dropna(subset=feats + [label])
    if len(tr) < MIN_TRAIN_ROWS:
        return np.zeros(len(valid_df))
    score = np.zeros(len(valid_df))
    V = valid_df[feats].fillna(0.0)
    for f in feats:
        ic = _rank_ic(tr, f, label)
        sgn = 0.0 if np.isnan(ic) else float(np.sign(ic))
        score += sgn * V[f].to_numpy()
    return score


def _fit_predict_one(kind: str, target: str, train_df, valid_df, feats, seed=None) -> np.ndarray:
    seed = SEED if seed is None else seed
    tr = train_df.dropna(subset=[target]).copy()
    if len(tr) < MIN_TRAIN_ROWS:
        return np.zeros(len(valid_df))
    tr = tr.sort_values("date")
    model = _new_model(kind, seed)
    Xtr, Xva = _X(tr, feats), _X(valid_df, feats)
    if kind == "rank_lgb":
        y = np.clip((tr[target].to_numpy() * RANK_BUCKETS).astype(int), 0, RANK_BUCKETS - 1)
        model.fit(Xtr, y, group=_group_sizes(tr))
    else:
        model.fit(Xtr, tr[target].to_numpy(dtype=float))
    return np.asarray(model.predict(Xva), dtype=float)


def stack_rank_mean(train_df, valid_df, feats, label):
    """A7 堆叠臂: 5 基树模型截面秩均值。"""
    ranks = []
    for _, target, kind in BASE_SPECS:
        pred = _fit_predict_one(kind, target, train_df, valid_df, feats)
        ranks.append(_xs_rank(valid_df, pred))
    return np.mean(np.vstack(ranks), axis=0)


def m14_bottom20_score(train_df, valid_df, feats, label):
    """M14 避雷: 预测 P(label_rank <= bottom20%), 分数越高越安全。"""
    tr = train_df.dropna(subset=["label_rank"]).copy()
    if len(tr) < MIN_TRAIN_ROWS:
        return np.zeros(len(valid_df))
    y = (tr["label_rank"].to_numpy(dtype=float) <= 0.20).astype(int)
    if y.sum() < 20 or y.sum() == len(y):
        return np.zeros(len(valid_df))
    p = dict(cfg.LGB_PARAMS)
    p["random_state"] = SEED
    clf = LGBMClassifier(**p)
    clf.fit(_X(tr, feats), y)
    # 返回安全分 = 1 - P(bottom20)
    prob = clf.predict_proba(_X(valid_df, feats))[:, 1]
    return 1.0 - np.asarray(prob, dtype=float)


def a7_blend_score(train_df, valid_df, feats, label, beta=None):
    """A7 = beta*rank(rule) + (1-beta)*rank(stack)。"""
    beta = cfg.A7_BETA if beta is None else beta
    rule = rule_score(train_df, valid_df, cfg.REV_FACTORS, label)
    stack = stack_rank_mean(train_df, valid_df, feats, label)
    r_rule = _xs_rank(valid_df, rule)
    r_stack = _xs_rank(valid_df, stack)
    return beta * r_rule + (1.0 - beta) * r_stack
