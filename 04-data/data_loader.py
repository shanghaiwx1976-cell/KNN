#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIX可复用数据加载器
提供清洗后的数据、技术指标特征和标签，供ML建模直接使用。
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent
RAW_CSV = DATA_DIR.parent / "SH500ETFVIX.csv"


class VIXDataLoader:
    """中证500 VIX数据加载和特征工程工具"""

    def __init__(self, csv_path=None):
        self.csv_path = csv_path or str(RAW_CSV)
        self._df = None
        self._features = None
        self._labels = None

    # ── 原始数据加载 ──────────────────────────────────────────────

    def load_raw(self) -> pd.DataFrame:
        """加载原始OHLC数据"""
        df = pd.read_csv(
            self.csv_path,
            header=None,
            names=["datetime", "open", "high", "low", "close"],
        )
        df["date"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    # ── 清洗数据 ─────────────────────────────────────────────────

    def get_clean_data(self) -> pd.DataFrame:
        """返回清洗后的数据（带基础衍生列）"""
        if self._df is not None:
            return self._df.copy()

        df = self.load_raw()

        # 基础价格衍生
        df["intraday_return"] = (df["close"] - df["open"]) / df["open"]
        df["intraday_range"] = (df["high"] - df["low"]) / df["open"]
        df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
        df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]
        df["close_to_close_return"] = df["close"].pct_change()
        df["open_to_open_return"] = df["open"].pct_change()
        df["overnight_gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        # 均线系列
        for w in [5, 10, 20, 60]:
            df[f"ma{w}"] = df["close"].rolling(w).mean()
            df[f"deviation_ma{w}"] = (df["close"] - df[f"ma{w}"]) / df[f"ma{w}"]

        # 波动率
        for w in [5, 10, 20, 60]:
            df[f"volatility_{w}d"] = df["close_to_close_return"].rolling(w).std()
            df[f"realized_vol_{w}d"] = df["intraday_range"].rolling(w).mean()

        # 分位数
        for w in [20, 60]:
            df[f"quantile_{w}d_30"] = df["close"].rolling(w).quantile(0.3)
            df[f"quantile_{w}d_50"] = df["close"].rolling(w).quantile(0.5)
            df[f"quantile_{w}d_70"] = df["close"].rolling(w).quantile(0.7)
            df[f"quantile_{w}d_90"] = df["close"].rolling(w).quantile(0.9)
            df[f"pctrank_{w}d"] = df["close"].rolling(w).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )

        # Z-score
        for w in [20, 60]:
            roll_mean = df["close"].rolling(w).mean()
            roll_std = df["close"].rolling(w).std()
            df[f"zscore_{w}d"] = (df["close"] - roll_mean) / roll_std

        # RSI
        for w in [6, 14]:
            delta = df["close"].diff()
            gain = delta.clip(lower=0).rolling(w).mean()
            loss = (-delta.clip(upper=0)).rolling(w).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f"rsi_{w}"] = 100 - 100 / (1 + rs)

        # 成交量代理 (用日内振幅作为活跃度代理)
        df["activity_proxy"] = df["intraday_range"] * df["close"]

        # 星期几 (one-hot编码用)
        df["weekday"] = df["date"].dt.weekday

        # 月份
        df["month"] = df["date"].dt.month

        self._df = df
        return df.copy()

    # ── 特征矩阵 ─────────────────────────────────────────────────

    def get_features(self) -> pd.DataFrame:
        """返回ML可用的特征矩阵"""
        if self._features is not None:
            return self._features.copy()

        df = self.get_clean_data()

        feature_cols = [
            # 价格特征
            "intraday_return", "intraday_range", "upper_shadow", "lower_shadow",
            "close_to_close_return", "open_to_open_return", "overnight_gap",
            # 均线偏离
            "deviation_ma5", "deviation_ma10", "deviation_ma20", "deviation_ma60",
            # 波动率
            "volatility_5d", "volatility_10d", "volatility_20d", "volatility_60d",
            "realized_vol_5d", "realized_vol_10d", "realized_vol_20d", "realized_vol_60d",
            # 分位数
            "pctrank_20d", "pctrank_60d",
            # Z-score
            "zscore_20d", "zscore_60d",
            # RSI
            "rsi_6", "rsi_14",
            # 活跃度
            "activity_proxy",
            # 时间
            "weekday", "month",
        ]

        # 滞后特征 (前1-5天)
        lag_cols = ["intraday_return", "close_to_close_return", "overnight_gap",
                    "intraday_range", "zscore_20d"]
        for col in lag_cols:
            for lag in range(1, 6):
                feat_name = f"{col}_lag{lag}"
                df[feat_name] = df[col].shift(lag)
                feature_cols.append(feat_name)

        # 滚动统计 (日内收益率)
        for w in [3, 5, 10]:
            df[f"intraday_return_mean_{w}d"] = df["intraday_return"].rolling(w).mean()
            df[f"intraday_return_std_{w}d"] = df["intraday_return"].rolling(w).std()
            df[f"win_rate_{w}d"] = df["intraday_return"].rolling(w).apply(
                lambda x: (x > 0).mean(), raw=True
            )
            feature_cols.extend([
                f"intraday_return_mean_{w}d",
                f"intraday_return_std_{w}d",
                f"win_rate_{w}d",
            ])

        # 连续涨跌天数
        df["streak"] = 0
        for i in range(1, len(df)):
            if df.iloc[i]["intraday_return"] * df.iloc[i - 1]["intraday_return"] > 0:
                df.iloc[i, df.columns.get_loc("streak")] = (
                    df.iloc[i - 1]["streak"]
                    + (1 if df.iloc[i]["intraday_return"] > 0 else -1)
                )
            else:
                df.iloc[i, df.columns.get_loc("streak")] = (
                    1 if df.iloc[i]["intraday_return"] > 0 else -1
                )
        feature_cols.append("streak")

        self._features = df[["date"] + feature_cols].copy()
        return self._features.copy()

    # ── 标签 ──────────────────────────────────────────────────────

    def get_labels(self) -> pd.DataFrame:
        """
        返回多种标签定义
        - intraday_direction: 次日日内方向 (1=收盘<开盘, 0=收盘>=开盘) → 做空是否盈利
        - intraday_return_next: 次日日内收益率 (open-close)/open
        - intraday_big_drop: 次日日内跌幅>2% (1/0)
        - close_direction_next: 次日收盘相比今日收盘方向
        - regime: 波动率状态 (high/medium/low 按分位数)
        """
        df = self.get_clean_data()

        labels = pd.DataFrame()
        labels["date"] = df["date"]

        # 次日做空是否盈利: (open-close)/open > 0 即做空赚钱
        next_intraday = ((df["open"].shift(-1) - df["close"].shift(-1))
                         / df["open"].shift(-1))
        labels["intraday_direction"] = (next_intraday > 0).astype(int)
        labels["intraday_return_next"] = next_intraday
        labels["intraday_big_drop"] = (next_intraday > 0.02).astype(int)

        # 次日收盘方向
        labels["close_direction_next"] = (df["close"].shift(-1) < df["close"]).astype(int)

        # 波动率状态
        q33 = df["close"].quantile(0.33)
        q67 = df["close"].quantile(0.67)
        labels["regime"] = pd.cut(
            df["close"], bins=[0, q33, q67, 999], labels=["low", "medium", "high"]
        )

        self._labels = labels
        return labels.copy()

    # ── 一站式ML数据集 ───────────────────────────────────────────

    def get_ml_dataset(self, label_type="intraday_direction", dropna=True):
        """
        获取 (X, y) 形式的ML数据集

        Parameters
        ----------
        label_type : str
            标签列名, 可选:
            'intraday_direction', 'intraday_return_next',
            'intraday_big_drop', 'close_direction_next', 'regime'
        dropna : bool
            是否删除含NaN的行

        Returns
        -------
        X : pd.DataFrame  特征矩阵
        y : pd.Series      标签
        """
        features = self.get_features()
        labels = self.get_labels()
        merged = features.merge(labels[["date", label_type]], on="date")

        if dropna:
            merged = merged.dropna().reset_index(drop=True)

        feature_cols = [c for c in features.columns if c != "date"]
        X = merged[feature_cols]
        y = merged[label_type]
        return X, y

    def train_test_split(self, label_type="intraday_direction",
                         test_ratio=0.2, dropna=True):
        """
        时序切分 (前 1-test_ratio 为训练集, 后 test_ratio 为测试集)
        """
        X, y = self.get_ml_dataset(label_type=label_type, dropna=dropna)
        n = len(X)
        split = int(n * (1 - test_ratio))
        return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    # ── 导出 ──────────────────────────────────────────────────────

    def export_all(self, output_dir=None):
        """导出所有整理后的数据到CSV"""
        output_dir = Path(output_dir) if output_dir else DATA_DIR

        clean = self.get_clean_data()
        clean.to_csv(output_dir / "SH500ETFVIX_clean.csv", index=False, encoding="utf-8-sig")

        features = self.get_features()
        features.to_csv(output_dir / "features_base.csv", index=False, encoding="utf-8-sig")

        labels = self.get_labels()
        labels.to_csv(output_dir / "labels.csv", index=False, encoding="utf-8-sig")

        print(f"数据已导出到 {output_dir}:")
        print(f"  - SH500ETFVIX_clean.csv  ({len(clean)} 行, {len(clean.columns)} 列)")
        print(f"  - features_base.csv      ({len(features)} 行, {len(features.columns)} 列)")
        print(f"  - labels.csv             ({len(labels)} 行, {len(labels.columns)} 列)")


if __name__ == "__main__":
    loader = VIXDataLoader()
    loader.export_all()

    # 验证ML数据集
    X, y = loader.get_ml_dataset()
    print(f"\nML数据集: X={X.shape}, y={y.shape}")
    print(f"标签分布: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = loader.train_test_split()
    print(f"训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集: X={X_test.shape}, y={y_test.shape}")
