# THU-BDC2026 Alpha Strategy Optimization

## Strategy: A7_TOP2_DD10

Optimized configuration for the Tsinghua Big Data Competition 2026 (stock alpha prediction).

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Alpha | A7 (5-model blend) | LGB_reg, LGB_rank, RF, ET, HistGB |
| TOP_K | 2 | Extreme concentration (top 2 stocks) |
| SUBPOOL_FRAC | 1/3 | High-turnover subpool filter |
| DD_THRESHOLD | -10% | Drawdown protection trigger |
| DD_REDUCE | 30% | Position reduction on trigger |

### OOS Performance (252 trading days)

| Metric | Value |
|--------|-------|
| Net Return | **3.383** |
| Sharpe Ratio | 5.88 |
| Max Drawdown | -20.5% |
| Win Rate | 62.3% |
| P(Top 10%) | **94.8%** |

### Time-Series Consistency

- 4-fold cross-validation: **100%** (all folds positive)
- Minimum fold return: +22.4%
- Half-split validation: both halves positive

### Architecture

```
Features (31 factors)
  → 5 ML Models (walk-forward OOS)
    → Rank-average blend (A7)
      → Subpool filter (top 1/3 turnover)
        → Buffer selection (top 2, hysteresis)
          → Drawdown protection (prev-day signal, no look-ahead)
            → Final positions
```

### Files

- `code/src/config.py` - All configurable parameters
- `code/src/alpha_pipeline.py` - Core pipeline with drawdown protection
- `code/src/alpha_scorers_v2.py` - 5-model ensemble scorer
- `code/src/alpha_features_v2.py` - 31 alpha factors
- `code/src/run_final_v3.py` - Optimization script (140 configs tested)
- `code/src/train.py` - Training and OOS evaluation
- `code/src/predict.py` - Daily prediction entry point
