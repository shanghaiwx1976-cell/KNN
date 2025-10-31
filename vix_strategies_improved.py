#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中证500 VIX波动率做空策略回测系统 - 改进版
包含Sharpe比率计算和动态仓位管理
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class VIXShortStrategiesImproved:
    """VIX做空策略改进类 - 包含仓位管理"""

    def __init__(self, csv_file, risk_free_rate=0.025):
        """初始化并加载数据"""
        self.df = self.load_data(csv_file)
        self.risk_free_rate = risk_free_rate  # 无风险利率（年化2.5%）
        self.results = {}
        self.results_improved = {}

    def load_data(self, csv_file):
        """加载CSV数据"""
        df = pd.read_csv(csv_file, header=None,
                        names=['datetime', 'open', 'high', 'low', 'close'])
        df['date'] = pd.to_datetime(df['datetime'])
        return df

    def calculate_sharpe_ratio(self, returns, trading_days=252):
        """
        计算Sharpe比率
        returns: 日收益率序列
        trading_days: 年化交易日数
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0

        # 计算年化收益率
        annual_return = returns.mean() * trading_days

        # 计算年化波动率
        annual_volatility = returns.std() * np.sqrt(trading_days)

        # Sharpe比率 = (年化收益率 - 无风险利率) / 年化波动率
        sharpe = (annual_return - self.risk_free_rate) / annual_volatility

        return sharpe

    def calculate_metrics(self, df):
        """计算策略绩效指标"""
        # 总收益率
        total_return = df['cumulative_return'].iloc[-1] - 1

        # 年化收益率
        days = len(df)
        years = days / 252
        annual_return = (df['cumulative_return'].iloc[-1] ** (1/years)) - 1

        # Sharpe比率
        sharpe = self.calculate_sharpe_ratio(df['daily_return'])

        # 最大回撤
        cumulative = df['cumulative_return']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # 持仓统计
        holding_days = df['position'].sum()
        total_days = len(df)
        holding_ratio = holding_days / total_days

        # 胜率
        winning_days = (df['daily_return'] > 0).sum()
        trading_days = (df['position'] > 0).sum()
        win_rate = winning_days / trading_days if trading_days > 0 else 0

        # 平均盈亏比
        avg_win = df[df['daily_return'] > 0]['daily_return'].mean()
        avg_loss = abs(df[df['daily_return'] < 0]['daily_return'].mean())
        profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'holding_days': holding_days,
            'total_days': total_days,
            'holding_ratio': holding_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio
        }

    # ==================== 原始策略（无仓位管理）====================

    def strategy1_mean_reversion(self):
        """策略1：均值回归策略（原始版）"""
        df = self.df.copy()
        window = 20

        df['ma20'] = df['close'].rolling(window=window).mean()
        df['signal'] = 0
        df.loc[df['close'] > df['ma20'], 'signal'] = 1
        df['position'] = df['signal'].shift(1).fillna(0)

        df['daily_return'] = 0.0
        mask = df['position'] == 1
        df.loc[mask, 'daily_return'] = (df.loc[mask, 'open'] - df.loc[mask, 'close']) / df.loc[mask, 'open']
        df['cumulative_return'] = (1 + df['daily_return']).cumprod()

        metrics = self.calculate_metrics(df)
        metrics['name'] = '策略1: 均值回归（原始）'
        metrics['df'] = df
        return metrics

    def strategy2_quantile_short(self):
        """策略2：分位数高位做空策略（原始版）"""
        df = self.df.copy()
        window = 60

        df['q50'] = df['close'].rolling(window=window).quantile(0.50)
        df['q70'] = df['close'].rolling(window=window).quantile(0.70)
        df['signal'] = 0

        for i in range(window, len(df)):
            current_close = df.loc[i, 'close']
            q50 = df.loc[i, 'q50']
            q70 = df.loc[i, 'q70']

            if current_close > q70:
                df.loc[i, 'signal'] = 1
            elif current_close < q50:
                df.loc[i, 'signal'] = 0
            else:
                if i > window:
                    df.loc[i, 'signal'] = df.loc[i-1, 'signal']

        df['position'] = df['signal'].shift(1).fillna(0)
        df['daily_return'] = 0.0
        mask = df['position'] == 1
        df.loc[mask, 'daily_return'] = (df.loc[mask, 'open'] - df.loc[mask, 'close']) / df.loc[mask, 'open']
        df['cumulative_return'] = (1 + df['daily_return']).cumprod()

        metrics = self.calculate_metrics(df)
        metrics['name'] = '策略2: 分位数（原始）'
        metrics['df'] = df
        return metrics

    def strategy3_daily_intraday(self):
        """策略3：日内差价策略（原始版）"""
        df = self.df.copy()
        df['position'] = 1
        df['daily_return'] = (df['open'] - df['close']) / df['open']
        df['cumulative_return'] = (1 + df['daily_return']).cumprod()

        metrics = self.calculate_metrics(df)
        metrics['name'] = '策略3: 日内差价（原始）'
        metrics['df'] = df
        return metrics

    # ==================== 改进策略（带仓位管理）====================

    def strategy1_improved(self):
        """
        策略1改进版：基于偏离度的动态仓位管理
        - 仓位 = 基础仓位 * 偏离度系数
        - 偏离度 = (VIX - MA20) / MA20
        - 偏离度越大，仓位越大
        """
        df = self.df.copy()
        window = 20

        df['ma20'] = df['close'].rolling(window=window).mean()

        # 计算偏离度
        df['deviation'] = (df['close'] - df['ma20']) / df['ma20']

        # 动态仓位：基础仓位0.5，根据偏离度调整
        # 偏离度 > 0: 做空信号，仓位随偏离度增加
        # 偏离度 <= 0: 不做空
        df['position'] = 0.0

        for i in range(window, len(df)):
            deviation = df.loc[i-1, 'deviation']  # 使用前一日数据

            if deviation > 0:
                # 仓位 = 0.5 + 偏离度 * 5（最大仓位100%）
                position_size = min(0.5 + deviation * 5, 1.0)
                df.loc[i, 'position'] = position_size
            else:
                df.loc[i, 'position'] = 0.0

        # 计算收益（考虑仓位大小）
        df['daily_return'] = 0.0
        mask = df['position'] > 0
        df.loc[mask, 'daily_return'] = df.loc[mask, 'position'] * \
                                        (df.loc[mask, 'open'] - df.loc[mask, 'close']) / df.loc[mask, 'open']
        df['cumulative_return'] = (1 + df['daily_return']).cumprod()

        metrics = self.calculate_metrics(df)
        metrics['name'] = '策略1: 均值回归（仓位管理）'
        metrics['df'] = df
        return metrics

    def strategy2_improved(self):
        """
        策略2改进版：基于分位数位置的动态仓位管理
        - 仓位与VIX在分位数中的位置成正比
        - VIX越接近100%分位数，仓位越大
        """
        df = self.df.copy()
        window = 60

        df['q30'] = df['close'].rolling(window=window).quantile(0.30)
        df['q50'] = df['close'].rolling(window=window).quantile(0.50)
        df['q70'] = df['close'].rolling(window=window).quantile(0.70)
        df['q90'] = df['close'].rolling(window=window).quantile(0.90)

        df['position'] = 0.0

        for i in range(window, len(df)):
            current_close = df.loc[i-1, 'close']  # 使用前一日数据
            q30 = df.loc[i-1, 'q30']
            q50 = df.loc[i-1, 'q50']
            q70 = df.loc[i-1, 'q70']
            q90 = df.loc[i-1, 'q90']

            # 动态仓位分配
            if current_close > q90:
                # 极高位：满仓做空
                df.loc[i, 'position'] = 1.0
            elif current_close > q70:
                # 高位：70%仓位
                df.loc[i, 'position'] = 0.7
            elif current_close > q50:
                # 中高位：40%仓位
                df.loc[i, 'position'] = 0.4
            elif current_close > q30:
                # 中低位：20%仓位
                df.loc[i, 'position'] = 0.2
            else:
                # 低位：空仓
                df.loc[i, 'position'] = 0.0

        # 计算收益
        df['daily_return'] = 0.0
        mask = df['position'] > 0
        df.loc[mask, 'daily_return'] = df.loc[mask, 'position'] * \
                                        (df.loc[mask, 'open'] - df.loc[mask, 'close']) / df.loc[mask, 'open']
        df['cumulative_return'] = (1 + df['daily_return']).cumprod()

        metrics = self.calculate_metrics(df)
        metrics['name'] = '策略2: 分位数（仓位管理）'
        metrics['df'] = df
        return metrics

    def strategy3_improved(self):
        """
        策略3改进版：基于VIX绝对水平的动态仓位管理
        - 根据VIX绝对值大小调整仓位
        - VIX越高，仓位越大
        """
        df = self.df.copy()

        # 计算VIX的历史滚动统计
        df['vix_mean'] = df['close'].rolling(window=60).mean()
        df['vix_std'] = df['close'].rolling(window=60).std()

        df['position'] = 0.0

        for i in range(60, len(df)):
            current_vix = df.loc[i, 'close']
            vix_mean = df.loc[i, 'vix_mean']
            vix_std = df.loc[i, 'vix_std']

            # 计算Z-score（标准化后的VIX水平）
            if vix_std > 0:
                z_score = (current_vix - vix_mean) / vix_std

                # 根据Z-score确定仓位
                if z_score > 2.0:
                    # VIX极高（超过2个标准差）：满仓
                    position_size = 1.0
                elif z_score > 1.0:
                    # VIX很高：80%仓位
                    position_size = 0.8
                elif z_score > 0.5:
                    # VIX偏高：60%仓位
                    position_size = 0.6
                elif z_score > 0:
                    # VIX略高：40%仓位
                    position_size = 0.4
                elif z_score > -0.5:
                    # VIX正常：20%仓位
                    position_size = 0.2
                else:
                    # VIX偏低：不做空
                    position_size = 0.0
            else:
                position_size = 0.5  # 默认仓位

            df.loc[i, 'position'] = position_size

        # 计算收益
        df['daily_return'] = 0.0
        mask = df['position'] > 0
        df.loc[mask, 'daily_return'] = df.loc[mask, 'position'] * \
                                        (df.loc[mask, 'open'] - df.loc[mask, 'close']) / df.loc[mask, 'open']
        df['cumulative_return'] = (1 + df['daily_return']).cumprod()

        metrics = self.calculate_metrics(df)
        metrics['name'] = '策略3: 日内差价（仓位管理）'
        metrics['df'] = df
        return metrics

    def run_all_strategies(self):
        """运行所有策略（原始版和改进版）"""
        print("=" * 100)
        print("中证500 VIX波动率做空策略回测系统 - 包含Sharpe比率和仓位管理")
        print("=" * 100)
        print(f"\n数据期间: {self.df['date'].min()} 至 {self.df['date'].max()}")
        print(f"总交易日: {len(self.df)} 天")
        print(f"无风险利率: {self.risk_free_rate:.2%} (年化)\n")

        # 运行原始策略
        print("\n" + "=" * 100)
        print("【原始策略】- 无仓位管理")
        print("=" * 100)

        self.results['strategy1'] = self.strategy1_mean_reversion()
        self.results['strategy2'] = self.strategy2_quantile_short()
        self.results['strategy3'] = self.strategy3_daily_intraday()

        # 运行改进策略
        print("\n" + "=" * 100)
        print("【改进策略】- 带动态仓位管理")
        print("=" * 100)

        self.results_improved['strategy1'] = self.strategy1_improved()
        self.results_improved['strategy2'] = self.strategy2_improved()
        self.results_improved['strategy3'] = self.strategy3_improved()

        # 打印结果
        self.print_comparison()

        # 绘图
        self.plot_results()

        return self.results, self.results_improved

    def print_comparison(self):
        """打印原始策略与改进策略的对比"""
        print("\n" + "=" * 100)
        print("策略对比：原始版 vs 改进版（带仓位管理）")
        print("=" * 100)

        for key in ['strategy1', 'strategy2', 'strategy3']:
            original = self.results[key]
            improved = self.results_improved[key]

            print(f"\n{'='*100}")
            print(f"【{key.upper()}】")
            print(f"{'='*100}")

            # 原始策略
            print(f"\n  原始策略: {original['name']}")
            print(f"    总收益率:     {original['total_return']:>10.2%}")
            print(f"    年化收益率:   {original['annual_return']:>10.2%}")
            print(f"    Sharpe比率:   {original['sharpe_ratio']:>10.2f}")
            print(f"    最大回撤:     {original['max_drawdown']:>10.2%}")
            print(f"    持仓比例:     {original['holding_ratio']:>10.2%}")
            print(f"    胜率:         {original['win_rate']:>10.2%}")
            print(f"    盈亏比:       {original['profit_loss_ratio']:>10.2f}")

            # 改进策略
            print(f"\n  改进策略: {improved['name']}")
            print(f"    总收益率:     {improved['total_return']:>10.2%}")
            print(f"    年化收益率:   {improved['annual_return']:>10.2%}")
            print(f"    Sharpe比率:   {improved['sharpe_ratio']:>10.2f}")
            print(f"    最大回撤:     {improved['max_drawdown']:>10.2%}")
            print(f"    持仓比例:     {improved['holding_ratio']:>10.2%}")
            print(f"    胜率:         {improved['win_rate']:>10.2%}")
            print(f"    盈亏比:       {improved['profit_loss_ratio']:>10.2f}")

            # 改进效果
            print(f"\n  改进效果:")
            sharpe_improve = improved['sharpe_ratio'] - original['sharpe_ratio']
            drawdown_improve = improved['max_drawdown'] - original['max_drawdown']

            print(f"    Sharpe提升:   {sharpe_improve:>10.2f} {'✓' if sharpe_improve > 0 else '✗'}")
            print(f"    回撤改善:     {drawdown_improve:>10.2%} {'✓' if drawdown_improve > 0 else '✗'}")

    def plot_results(self):
        """绘制对比图表"""
        fig, axes = plt.subplots(3, 2, figsize=(18, 14))

        for idx, (key, original) in enumerate(self.results.items()):
            improved = self.results_improved[key]

            # 左列：原始策略
            ax_left = axes[idx, 0]
            df_orig = original['df']
            ax_left.plot(df_orig['date'], df_orig['cumulative_return'],
                        label=original['name'], color='#2E86AB', linewidth=2)
            ax_left.axhline(y=1, color='black', linestyle='--', alpha=0.3)
            ax_left.set_title(
                f"{original['name']}\n"
                f"收益: {original['total_return']:.1%} | "
                f"Sharpe: {original['sharpe_ratio']:.2f} | "
                f"回撤: {original['max_drawdown']:.1%}",
                fontsize=11, fontweight='bold'
            )
            ax_left.set_xlabel('日期')
            ax_left.set_ylabel('累积收益倍数')
            ax_left.legend()
            ax_left.grid(True, alpha=0.3)

            # 右列：改进策略
            ax_right = axes[idx, 1]
            df_impr = improved['df']
            ax_right.plot(df_impr['date'], df_impr['cumulative_return'],
                         label=improved['name'], color='#F18F01', linewidth=2)
            ax_right.axhline(y=1, color='black', linestyle='--', alpha=0.3)
            ax_right.set_title(
                f"{improved['name']}\n"
                f"收益: {improved['total_return']:.1%} | "
                f"Sharpe: {improved['sharpe_ratio']:.2f} | "
                f"回撤: {improved['max_drawdown']:.1%}",
                fontsize=11, fontweight='bold'
            )
            ax_right.set_xlabel('日期')
            ax_right.set_ylabel('累积收益倍数')
            ax_right.legend()
            ax_right.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/home/user/KNN/vix_strategies_comparison.png',
                    dpi=300, bbox_inches='tight')
        print("\n对比图表已保存至: /home/user/KNN/vix_strategies_comparison.png")

    def export_results(self):
        """导出详细结果"""
        for key, result in self.results_improved.items():
            filename = f"/home/user/KNN/{key}_improved_details.csv"
            df = result['df'][['date', 'open', 'high', 'low', 'close',
                              'position', 'daily_return', 'cumulative_return']]
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"改进策略详细数据已导出: {filename}")


def main():
    """主函数"""
    vix_strategies = VIXShortStrategiesImproved('/home/user/KNN/SH500ETFVIX.csv')

    # 运行所有策略
    results_original, results_improved = vix_strategies.run_all_strategies()

    # 导出结果
    print("\n" + "=" * 100)
    vix_strategies.export_results()
    print("=" * 100)

    return results_original, results_improved


if __name__ == "__main__":
    main()
