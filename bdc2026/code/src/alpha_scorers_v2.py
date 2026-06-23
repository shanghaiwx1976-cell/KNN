# -*- coding: utf-8 -*-
"""改进打分器 v2: Ridge元学习器 + 多种度量集成 + Purged K-Fold"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge

import config as cfg

SEED = cfg.SEED
RANK_BUCKETS = cfg.LGB_RANK_BUCKETS
MIN_TRAIN_ROWS = cfg.MIN_TRAIN_ROWS


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
    if kind == "reg_lgb_deep":
        p = dict(cfg.LGB_PARAMS)
        p["random_state"] = seed
        p["max_depth"] = 7
        p["num_leaves"] = 63
        p["n_estimators"] = 500
        p["learning_rate"] = 0.02
        return LGBMRegressor(**p)
    if kind == "reg_lgb_wide":
        p = dict(cfg.LGB_PARAMS)
        p["random_state"] = seed
        p["max_depth"] = 3
        p["num_leaves"] = 8
        p["n_estimators"] = 800
        p["learning_rate"] = 0.01
        p["subsample"] = 0.6
        return LGBMRegressor(**p)
    raise ValueError(kind)


# Extended base models
BASE_SPECS_V2 = [
    ("lgbreg", "label", "reg_lgb"),
    ("lgbrank", "label_rank", "rank_lgb"),
    ("histgb", "label", "reg_histgb"),
    ("rf", "label", "reg_rf"),
    ("et", "label", "reg_et"),
    ("lgb_deep", "label", "reg_lgb_deep"),
    ("lgb_wide", "label", "reg_lgb_wide"),
]


def rule_score_v2(train_df, valid_df, feats, label):
    """改进规则分: IC加权 (非等权) + 多期因子。"""
    tr = train_df.dropna(subset=feats + [label])
    if len(tr) < MIN_TRAIN_ROWS:
        return np.zeros(len(valid_df))
    V = valid_df[feats].fillna(0.0)
    score = np.zeros(len(valid_df))
    total_weight = 0.0
    for f in feats:
        ic = _rank_ic(tr, f, label)
        if np.isnan(ic) or abs(ic) < 0.005:
            continue
        w = ic  # IC-weighted (preserves sign)
        score += w * V[f].to_numpy()
        total_weight += abs(w)
    if total_weight > 0:
        score /= total_weight
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


def stack_ridge_meta(train_df, valid_df, feats, label):
    """Ridge meta-learner: 用最近验证窗口学习最优权重。"""
    tr = train_df.dropna(subset=[label]).copy()
    if len(tr) < MIN_TRAIN_ROWS:
        return np.zeros(len(valid_df))
    tr = tr.sort_values("date")
    dates = sorted(tr["date"].unique())
    n_dates = len(dates)
    meta_split = max(n_dates - 42, n_dates * 3 // 4)
    meta_tr_dates = dates[:meta_split]
    meta_va_dates = dates[meta_split:]
    meta_tr = tr[tr["date"].isin(meta_tr_dates)]
    meta_va = tr[tr["date"].isin(meta_va_dates)]
    if len(meta_tr) < MIN_TRAIN_ROWS or len(meta_va) < 100:
        return stack_rank_mean_v2(train_df, valid_df, feats, label)
    base_preds_meta = []
    base_preds_test = []
    for name, target, kind in BASE_SPECS_V2:
        pred_meta = _fit_predict_one(kind, target, meta_tr, meta_va, feats)
        pred_test = _fit_predict_one(kind, target, tr, valid_df, feats)
        base_preds_meta.append(_xs_rank(meta_va, pred_meta))
        base_preds_test.append(_xs_rank(valid_df, pred_test))
    X_meta = np.column_stack(base_preds_meta)
    y_meta = meta_va[label].to_numpy(dtype=float)
    X_test = np.column_stack(base_preds_test)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_meta, y_meta)
    return ridge.predict(X_test)


def stack_rank_mean_v2(train_df, valid_df, feats, label):
    """7基树模型截面秩均值 (扩展版)。"""
    ranks = []
    for _, target, kind in BASE_SPECS_V2:
        pred = _fit_predict_one(kind, target, train_df, valid_df, feats)
        ranks.append(_xs_rank(valid_df, pred))
    return np.mean(np.vstack(ranks), axis=0)


def multi_seed_stack(train_df, valid_df, feats, label, n_seeds=3):
    """多种子集成: 降低单次训练随机性。"""
    all_preds = []
    base_seed = SEED
    for s in range(n_seeds):
        ranks = []
        seed = base_seed + s * 100
        for _, target, kind in BASE_SPECS_V2[:5]:
            pred = _fit_predict_one(kind, target, train_df, valid_df, feats, seed=seed)
            ranks.append(_xs_rank(valid_df, pred))
        all_preds.append(np.mean(np.vstack(ranks), axis=0))
    return np.mean(np.vstack(all_preds), axis=0)


def a7_blend_score_v2(train_df, valid_df, feats, label, beta=None):
    """A7 v2 = beta*rank(IC-weighted rule) + (1-beta)*rank(ridge meta)。"""
    beta = cfg.A7_BETA if beta is None else beta
    rule = rule_score_v2(train_df, valid_df, cfg.REV_FACTORS, label)
    stack = stack_ridge_meta(train_df, valid_df, feats, label)
    r_rule = _xs_rank(valid_df, rule)
    r_stack = _xs_rank(valid_df, stack)
    return beta * r_rule + (1.0 - beta) * r_stack


def a7_ensemble_score(train_df, valid_df, feats, label, beta=None):
    """终极集成: 混合rule + ridge_meta + multi_seed + 7模型秩均。"""
    beta = cfg.A7_BETA if beta is None else beta
    rule = rule_score_v2(train_df, valid_df, cfg.REV_FACTORS, label)
    ridge = stack_ridge_meta(train_df, valid_df, feats, label)
    mean7 = stack_rank_mean_v2(train_df, valid_df, feats, label)
    multi = multi_seed_stack(train_df, valid_df, feats, label, n_seeds=2)
    r_rule = _xs_rank(valid_df, rule)
    r_ridge = _xs_rank(valid_df, ridge)
    r_mean7 = _xs_rank(valid_df, mean7)
    r_multi = _xs_rank(valid_df, multi)
    return (
        beta * r_rule
        + 0.3 * r_ridge
        + 0.1 * r_mean7
        + (1.0 - beta - 0.4) * r_multi
    )


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
    prob = clf.predict_proba(_X(valid_df, feats))[:, 1]
    return 1.0 - np.asarray(prob, dtype=float)
