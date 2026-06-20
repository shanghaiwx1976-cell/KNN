# 04-data: 可复用数据模块

本文件夹包含整理后的可复用数据，下次实验无需从零开始整理数据。

## 文件说明

| 文件名 | 说明 | 更新日期 |
|--------|------|---------|
| `SH500ETFVIX_clean.csv` | 清洗后的中证500 VIX日线数据 (OHLC + 技术指标) | 2026-06-21 |
| `features_base.csv` | 基础因子特征矩阵 (可直接用于ML建模) | 2026-06-21 |
| `labels.csv` | 标签数据 (多种标签定义) | 2026-06-21 |
| `data_loader.py` | 数据加载工具类 | 2026-06-21 |

## 快速使用

```python
from data_loader import VIXDataLoader

loader = VIXDataLoader()
X, y = loader.get_ml_dataset(label_type='intraday_direction')
X_train, X_test, y_train, y_test = loader.train_test_split(test_ratio=0.2)
```

## 数据时间范围

- 起始: 2022-09-19
- 结束: 2025-02-07
- 交易日数: 575
