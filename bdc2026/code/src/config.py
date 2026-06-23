# -*- coding: utf-8 -*-
"""WX ALPHA 策略配置 — A7 + 冠军M0 + M14 避雷"""

from pathlib import Path

# 路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = str(PROJECT_ROOT / "data")
OUTPUT_DIR = str(PROJECT_ROOT / "model" / "wx_a7_m14")

# 策略臂
STRATEGY = "A7_TOP2_DD10"  # A7 + top2集中 + 回撤保护

# 冠军 M0 子池: turn_5 高换手上 1/3
SUBPOOL_FRAC = 1.0 / 3.0
TOP_K = 2
WEIGHT = 0.5

# A7 混合系数 (预登记, 不调参)
A7_BETA = 0.5  # rule 权重; stack 权重 = 1 - A7_BETA

# M14 避雷: bottom20 分类器阈值 (P(bottom20) > thresh 则剔除)
M14_GATE_QUANTILE = 0.50  # 保留 rank_m14 >= 50% 分位 (避雷后幸存)

# 冠军机制
USE_BUFFER = True
BUFFER_KEEP = 8   # 4 * TOP_K
BUFFER_REPLACE = 12  # 6 * TOP_K
DEFAULT_TIMING = "T00"  # 送审轨道A: OOS net 最优

# 回撤保护 (基于前一日回撤信号, 无未来信息)
USE_DD_PROTECT = True
DD_THRESHOLD = -0.10   # 回撤超过10%时触发减仓
DD_REDUCE = 0.30       # 触发时仓位降至30%
DD_LOOKBACK = 20       # 回撤计算窗口 (交易日)

# Devin vol-target (轨道B: 设 USE_VOL_TARGET=True)
USE_VOL_TARGET = False
VOL_WINDOW = 13
VOL_TARGET = 0.028
VOL_CAP = 1.0

# 扩展数据训练窗 (若数据够长)
EXTENDED_TRAIN_START = "2018-01-01"

# 择时仓位阶梯 (三态策略用)
REGIME_LADDER = {"strong": 1.0, "neutral": 1.0, "weak": 0.25}

# 反转因子 (规则打分用)
REV_FACTORS = ["rev_5_std", "rev_10_std", "rev_20_std"]

# 全量 17 因子 (树模型用)
STD_FACTORS = [
    "rev_5_std", "rev_10_std", "rev_20_std", "mom_20_std",
    "turn_5_std", "turn_chg_std", "amihud_20_std",
    "vol_20_std", "dnvol_20_std", "idio_vol_20_std",
    "amp_5_std", "volr_5_std", "lowsh_5_std", "upsh_5_std",
    "rev_turn_std", "rev_vol_std", "rev_volr_std",
]

# LightGBM 超参 (强正则浅树)
LGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}
LGB_RANK_BUCKETS = 8
SEED = 20260416
MIN_TRAIN_ROWS = 500

# Walk-forward (交易日) — 适配赛制约2年数据
WF_TRAIN_DAYS = 252
WF_VALID_DAYS = 42
WF_STEP_DAYS = 42
EMBARGO_DAYS = 5
HOLDING_DAYS = 5

# 标签: open_t1 -> open_t5 收益 (与赛制一致)
LABEL_COL = "label"

config = {
    "strategy": STRATEGY,
    "data_path": DATA_PATH,
    "output_dir": OUTPUT_DIR,
    "top_k": TOP_K,
    "a7_beta": A7_BETA,
    "m14_gate_quantile": M14_GATE_QUANTILE,
    "subpool_frac": SUBPOOL_FRAC,
    "std_factors": STD_FACTORS,
    "rev_factors": REV_FACTORS,
    "seed": SEED,
}
