#!/usr/bin/env python3
"""Agent5: Temporal Feature Augmentation"""
from __future__ import annotations
import sys, warnings, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from lightgbm import LGBMRegressor, LGBMRanker
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
warnings.filterwarnings("ignore")

SEED = 20260416; CUTOFF = pd.Timestamp("2026-06-12")
SPLIT_KW = dict(train_days=104, valid_days=13, step_days=13, embargo_days=1, holding_days=1)
STD_FACTORS = ["rev_5_std","rev_10_std","rev_20_std","mom_20_std","turn_5_std","turn_chg_std","amihud_20_std","vol_20_std","dnvol_20_std","idio_vol_20_std","amp_5_std","volr_5_std","lowsh_5_std","upsh_5_std","rev_turn_std","rev_vol_std","rev_volr_std"]
REV_STD = ["rev_5_std","rev_10_std","rev_20_std"]
SUBPOOL_COL = "turn_5_std"; REGIME_FAST, REGIME_SLOW = 60, 150
LADDER = dict(strong=1.0, neutral=1.0, weak=0.25)
LGB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, min_child_samples=50, verbose=-1)
RANK_BUCKETS = 8; MIN_TRAIN_ROWS = 500
BASE_SPECS = [("lgbreg","label_excess","reg_lgb"),("lgbrank","label_rank","rank_lgb"),("histgb","label_excess","reg_histgb"),("rf","label_excess","reg_rf"),("et","label_excess","reg_et")]

def walk_forward_splits(dates, train_days, valid_days, step_days, embargo_days, holding_days):
    ud = sorted(dates.unique()); splits = []; i = 0
    while i+train_days+embargo_days+valid_days <= len(ud):
        splits.append((set(ud[i:i+train_days]), set(ud[i+train_days+embargo_days:i+train_days+embargo_days+valid_days]))); i += step_days
    return list(enumerate(splits))

def rank_ic(df, sc, lc):
    ics = []
    for d, g in df.groupby("date"):
        v = g[[sc,lc]].dropna()
        if len(v)>=10: c,_=spearmanr(v[sc],v[lc]); ics.append(c)
    return pd.Series(ics)

def select_topk_buffered(preds, k=5, keep_band=20, replace_band=30, tradable_col="tradable_entry", weighting="equal"):
    pos=[]; held=set()
    for d in sorted(preds["date"].unique()):
        g=preds[preds["date"]==d].copy()
        if tradable_col in g.columns:
            t=g[g[tradable_col]>0]
            if len(t)>=k: g=t
        ranked=g.sort_values("score",ascending=False); top=list(ranked["code"].values)
        keep=[c for c in held if c in top[:keep_band]]; need=k-len(keep)
        new=[c for c in top if c not in keep][:need]; sel=(keep+new)[:k]; held=set(sel)
        w=1.0/max(len(sel),1)
        for c in sel: pos.append({"date":d,"code":c,"weight":w})
    return pd.DataFrame(pos)

def compute_turnover(positions):
    to=[]; dates=sorted(positions["date"].unique()); prev=set()
    for d in dates:
        curr=set(positions[positions["date"]==d]["code"])
        to.append({"date":d,"buys":len(curr-prev),"sells":len(prev-curr)}); prev=curr
    return pd.DataFrame(to)

def backtest_simple(positions, preds, cost_rate=0.002):
    pos=positions.merge(preds[["date","code","fwd_ret"]], on=["date","code"], how="left")
    to=compute_turnover(positions); to_map=dict(zip(to["date"],to["buys"]))
    net_s,gross_s={},{}
    for d,g in pos.groupby("date"):
        ws=g["weight"].sum()
        if ws<=0: continue
        w=g["weight"]/ws; gr=float((w*g["fwd_ret"].fillna(0)).sum())
        nt=to_map.get(d,0); cost=cost_rate*nt/max(len(g),1)
        net_s[d]=gr-cost; gross_s[d]=gr
    net=pd.Series(net_s).sort_index(); gross=pd.Series(gross_s).sort_index()
    nt=float((1+net).prod()-1); cum=(1+net).cumprod(); dd=cum/cum.cummax()-1
    mdd=float(dd.min()) if len(dd) else 0
    sh=float(net.mean()/net.std()*np.sqrt(52)) if len(net)>1 and net.std()>0 else 0
    return {"net":net,"gross":gross,"summary":{"net_total":nt,"max_drawdown":mdd,"sharpe_annual":sh}}

def load_panel(fp, lp):
    f=pd.read_parquet(fp); l=pd.read_parquet(lp)
    f["date"]=pd.to_datetime(f["date"]); l["date"]=pd.to_datetime(l["date"])
    p=f.merge(l[["code","date","fwd_ret","label_excess","label_rank","bench_ret"]], on=["code","date"], how="inner")
    return p[p["date"]<=CUTOFF].sort_values(["date","code"]).reset_index(drop=True)

def build_regime_map(dates, hp, fast=REGIME_FAST, slow=REGIME_SLOW):
    idx=pd.read_parquet(hp).sort_values("date"); idx["date"]=pd.to_datetime(idx["date"])
    idx["close"]=pd.to_numeric(idx["close"],errors="coerce")
    idx["maf"]=idx["close"].rolling(fast,min_periods=fast//2).mean()
    idx["mas"]=idx["close"].rolling(slow,min_periods=slow//2).mean()
    d=pd.merge_asof(pd.DataFrame({"date":sorted(pd.to_datetime(list(dates)))}), idx[["date","close","maf","mas"]], on="date", direction="backward")
    def cls(r):
        if pd.isna(r["mas"]) or pd.isna(r["maf"]): return "neutral"
        if r["close"]>r["maf"] and r["close"]>r["mas"]: return "strong"
        if r["close"]<r["maf"] and r["close"]<r["mas"]: return "weak"
        return "neutral"
    d["reg"]=d.apply(cls,axis=1); return dict(zip(d["date"],d["reg"]))

def build_bench_map(panel):
    b=panel.dropna(subset=["bench_ret"]).groupby("date")["bench_ret"].first()
    return dict(zip(b.index,b.values))

def _X(df, feats): return df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
def _xs_rank(df, values):
    s=pd.Series(values, index=df.index); return df.assign(_v=s).groupby("date")["_v"].rank(pct=True).to_numpy()
def _group_sizes(df): return df.groupby("date", sort=True).size().tolist()

def _fit_predict_one(kind, target, train_df, valid_df, feats, seed=SEED):
    tr=train_df.dropna(subset=[target]).copy()
    if len(tr)<MIN_TRAIN_ROWS: return np.zeros(len(valid_df))
    tr=tr.sort_values("date"); Xtr=_X(tr,feats); Xva=_X(valid_df,feats)
    y=tr[target].to_numpy(dtype=float)
    if kind=="reg_lgb":
        p=dict(LGB_PARAMS); p["random_state"]=seed
        m=LGBMRegressor(**p); m.fit(Xtr,y)
    elif kind=="rank_lgb":
        p=dict(LGB_PARAMS); p["random_state"]=seed
        m=LGBMRanker(objective="lambdarank",**p)
        yr=np.clip((tr[target].to_numpy()*RANK_BUCKETS).astype(int),0,RANK_BUCKETS-1)
        m.fit(Xtr,yr,group=_group_sizes(tr))
    elif kind=="reg_histgb":
        m=HistGradientBoostingRegressor(max_depth=4,learning_rate=0.05,max_iter=300,l2_regularization=1.0,min_samples_leaf=50,random_state=seed)
        m.fit(Xtr,y)
    elif kind=="reg_rf":
        m=RandomForestRegressor(n_estimators=200,max_depth=6,min_samples_leaf=50,max_features="sqrt",n_jobs=-1,random_state=seed)
        m.fit(Xtr,y)
    elif kind=="reg_et":
        m=ExtraTreesRegressor(n_estimators=200,max_depth=6,min_samples_leaf=50,max_features="sqrt",n_jobs=-1,random_state=seed)
        m.fit(Xtr,y)
    else: raise ValueError(kind)
    return np.asarray(m.predict(Xva), dtype=float)

def rule_score(train_df, valid_df, feats, label):
    feats=list(feats); tr=train_df.dropna(subset=feats+[label])
    if len(tr)<MIN_TRAIN_ROWS: return np.zeros(len(valid_df))
    V=valid_df[feats].fillna(0.0); score=np.zeros(len(valid_df))
    for f in feats:
        ic=rank_ic(tr,f,label); m=ic.mean() if len(ic) else 0.0
        sgn=0.0 if (m is None or np.isnan(m)) else float(np.sign(m))
        score=score+sgn*V[f].to_numpy()
    return score

# === TEMPORAL FEATURE AUGMENTATION ===
def add_temporal_features(panel, factors, lag_periods=[1,2], add_delta=True, add_rollstd=False):
    """给面板添加滞后特征和变化率特征"""
    panel = panel.sort_values(["code", "date"]).copy()
    new_cols = []

    for f in factors:
        for lag in lag_periods:
            col = f"{f}_lag{lag}"
            panel[col] = panel.groupby("code")[f].shift(lag)
            new_cols.append(col)

        if add_delta:
            col = f"{f}_delta1"
            panel[col] = panel[f] - panel.groupby("code")[f].shift(1)
            new_cols.append(col)

        if add_rollstd:
            col = f"{f}_rstd2"
            panel[col] = panel.groupby("code")[f].rolling(2, min_periods=1).std().reset_index(0, drop=True)
            new_cols.append(col)

    # Fill NaN from lagging with 0 (z-score centered, so 0 = neutral)
    for c in new_cols:
        panel[c] = panel[c].fillna(0.0)

    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    return panel, new_cols

# Different feature set variants
def make_temporal_scorer(lag_periods, add_delta, add_rollstd, beta=0.5, use_factors=None):
    factors = use_factors or STD_FACTORS

    def scorer(train_df, valid_df, feats, label):
        # Add temporal features to both train and valid
        combined = pd.concat([train_df, valid_df], ignore_index=True)
        combined, new_cols = add_temporal_features(combined, factors, lag_periods, add_delta, add_rollstd)

        # Split back
        train_dates = set(train_df["date"].unique())
        valid_dates = set(valid_df["date"].unique())
        tr_aug = combined[combined["date"].isin(train_dates)].copy()
        va_aug = combined[combined["date"].isin(valid_dates)].copy()

        ext_feats = list(feats) + new_cols

        # Rule component (on original REV_STD only)
        r = rule_score(train_df, valid_df, REV_STD, label)

        # ML stack with extended features
        ranks = []
        for n, target, kind in BASE_SPECS:
            pred = _fit_predict_one(kind, target, tr_aug, va_aug, ext_feats)
            ranks.append(_xs_rank(va_aug, pred))
        s = np.mean(np.vstack(ranks), axis=0)

        return beta * _xs_rank(valid_df, r) + (1-beta) * _xs_rank(valid_df, s)
    return scorer

# Evaluation pipeline
def predict_oos(panel, scorer, feature_cols, label):
    splits=walk_forward_splits(panel["date"],**SPLIT_KW); parts=[]
    for wid,(tr,va) in splits:
        tr_df=panel[panel["date"].isin(tr)]; va_df=panel[panel["date"].isin(va)].copy()
        va_df["score"]=np.asarray(scorer(tr_df,va_df,feature_cols,label)); va_df["window"]=wid
        parts.append(va_df)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()

def filter_subpool(preds, col=SUBPOOL_COL, frac=1.0/3):
    out=[]
    for _,g in preds.groupby("date"):
        q=g[col].quantile(1-frac); out.append(g[g[col]>=q])
    return pd.concat(out,ignore_index=True)

def apply_regime_sizer(pos, rm, ladder=LADDER):
    p=pos.copy(); f=p["date"].map(lambda d: ladder.get(rm.get(d,"neutral"),ladder["neutral"]))
    p["weight"]=p["weight"]*f.to_numpy(); return p

def realized_dates(preds):
    v=preds.groupby("date")["fwd_ret"].apply(lambda s: s.notna().sum()>0)
    return {pd.Timestamp(d) for d,ok in v.items() if bool(ok)}

def evaluate(name, preds, rm, bm):
    sp=filter_subpool(preds); pos=select_topk_buffered(sp)
    if pos.empty: return {"strategy":name,"error":"no_positions","top_real_rank_pct":0,"net_total":0,"max_drawdown":0}
    pos=apply_regime_sizer(pos,rm)
    ed=realized_dates(preds)
    pos_e=pos[pos["date"].map(lambda d: pd.Timestamp(d) in ed)].copy()
    preds_e=preds[preds["date"].map(lambda d: pd.Timestamp(d) in ed)].copy()
    bt=backtest_simple(pos_e,preds_e); net=bt["net"].dropna()
    ic=rank_ic(preds_e,"score","label_excess")
    held=pos_e.merge(preds_e[["date","code","label_rank","fwd_ret"]],on=["date","code"],how="left")
    trp=float(held["label_rank"].mean()) if len(held) else np.nan
    ha5=float((held["label_rank"]>=0.8).mean()) if len(held) else np.nan
    s=bt["summary"]
    return {"strategy":name,"net_total":s["net_total"],"max_drawdown":s["max_drawdown"],
            "rank_ic_mean":float(ic.mean()) if len(ic.dropna()) else np.nan,
            "top_real_rank_pct":trp,"precision_at5":ha5}

# A7 baseline
def a7_scorer(train_df, valid_df, feats, label):
    r=rule_score(train_df,valid_df,REV_STD,label); ranks=[]
    for n,t,k in BASE_SPECS:
        pred=_fit_predict_one(k,t,train_df,valid_df,STD_FACTORS); ranks.append(_xs_rank(valid_df,pred))
    s=np.mean(np.vstack(ranks),axis=0)
    return 0.5*_xs_rank(valid_df,r)+0.5*_xs_rank(valid_df,s)

if __name__ == "__main__":
    DATA="/home/ubuntu/work/data_parquet"
    panel=load_panel(f"{DATA}/opus_feature_store_v1_weekly_20260612.parquet", f"{DATA}/opus_labels_weekly_20260612.parquet")
    rm=build_regime_map(panel["date"].unique(), f"{DATA}/hs300.parquet")
    bm=build_bench_map(panel)
    print(f"Panel: {panel.shape}")

    print("\n=== A7 Baseline ===")
    preds_a7=predict_oos(panel, a7_scorer, STD_FACTORS, "label_excess")
    r_a7=evaluate("A7_baseline",preds_a7,rm,bm)
    print(f"A7: IC={r_a7['top_real_rank_pct']:.4f}, net={r_a7['net_total']:.4f}, mdd={r_a7['max_drawdown']:.4f}")

    variants = [
        ("lag1_delta", make_temporal_scorer([1], True, False)),
        ("lag12_delta", make_temporal_scorer([1,2], True, False)),
        ("lag12_delta_rstd", make_temporal_scorer([1,2], True, True)),
        ("lag1_only", make_temporal_scorer([1], False, False)),
        ("lag12_delta_b04", make_temporal_scorer([1,2], True, False, beta=0.4)),
        ("lag12_delta_b03", make_temporal_scorer([1,2], True, False, beta=0.3)),
        # Only use key factors for temporal
        ("lag12_delta_keyfactors", make_temporal_scorer([1,2], True, False, use_factors=["rev_5_std","rev_10_std","rev_20_std","turn_5_std","vol_20_std","amihud_20_std"])),
    ]

    results=[r_a7]
    for name,scorer in variants:
        print(f"\n=== {name} ===")
        preds=predict_oos(panel,scorer,STD_FACTORS,"label_excess")
        r=evaluate(name,preds,rm,bm)
        print(f"{name}: IC={r['top_real_rank_pct']:.4f}, net={r['net_total']:.4f}, mdd={r['max_drawdown']:.4f}")
        results.append(r)

    print("\n"+"="*80)
    print("FINAL SUMMARY - Agent5 Temporal Features")
    print("="*80)
    results.sort(key=lambda x: x.get('top_real_rank_pct',0), reverse=True)
    for r in results:
        print(f"{r['strategy']:<30s} IC={r.get('top_real_rank_pct',0):.4f} net={r.get('net_total',0):.4f} mdd={r.get('max_drawdown',0):.4f}")
    best=results[0]
    print(f"\nBEST: {best['strategy']} IC={best['top_real_rank_pct']:.4f}")
