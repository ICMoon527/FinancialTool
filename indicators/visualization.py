import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, List, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class IndicatorVisualizer:
    """
    智牛特色指标可视化工具
    提供多种图表展示方式，包括K线图、折线图、柱状图、热力图等
    """

    def __init__(self, figsize: Tuple[int, int] = (16, 10)):
        """
        初始化可视化工具

        Args:
            figsize: 图表大小
        """
        self.figsize = figsize
        self.colors = {
            'red': '#FF4444',
            'green': '#44AA44',
            'blue': '#4477FF',
            'yellow': '#FFAA00',
            'purple': '#AA44FF',
            'orange': '#FF8800',
            'cyan': '#44DDDD',
            'gray': '#888888'
        }

    def plot_kline_with_indicator(
        self,
        data: pd.DataFrame,
        indicator_data: Optional[pd.DataFrame] = None,
        title: str = 'K线图与指标展示',
        main_indicator_columns: Optional[List[str]] = None,
        sub_indicator_columns: Optional[List[str]] = None,
        show_volume: bool = True
    ) -> plt.Figure:
        """
        绘制带有指标的K线图

        Args:
            data: OHLCV数据
            indicator_data: 指标数据
            title: 图表标题
            main_indicator_columns: 主图指标列名列表
            sub_indicator_columns: 副图指标列名列表
            show_volume: 是否显示成交量

        Returns:
            matplotlib Figure对象
        """
        n_rows = 1
        if show_volume:
            n_rows += 1
        if sub_indicator_columns and len(sub_indicator_columns) > 0:
            n_rows += 1

        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(n_rows, 1, height_ratios=[3, 1, 1][:n_rows])

        current_row = 0

        # 主图：K线 + 主图指标
        ax_main = fig.add_subplot(gs[current_row])
        self._plot_kline(ax_main, data)

        if main_indicator_columns and indicator_data is not None:
            for col in main_indicator_columns:
                if col in indicator_data.columns:
                    color = self.colors.get(col, self.colors['blue'])
                    ax_main.plot(indicator_data.index, indicator_data[col], 
                               label=col, color=color, linewidth=1.5, alpha=0.8)
            ax_main.legend(loc='upper left', fontsize=10)

        ax_main.set_title(title, fontsize=16, fontweight='bold')
        ax_main.grid(True, alpha=0.3)
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45)

        current_row += 1

        # 成交量
        if show_volume:
            ax_volume = fig.add_subplot(gs[current_row], sharex=ax_main)
            self._plot_volume(ax_volume, data)
            current_row += 1

        # 副图指标
        if sub_indicator_columns and indicator_data is not None:
            ax_sub = fig.add_subplot(gs[current_row], sharex=ax_main)
            for col in sub_indicator_columns:
                if col in indicator_data.columns:
                    color = self.colors.get(col, self.colors['purple'])
                    # 判断是柱状图还是折线图
                    if indicator_data[col].dtype == 'bool' or len(indicator_data[col].unique()) <= 10:
                        ax_sub.bar(indicator_data.index, indicator_data[col].astype(int), 
                                  label=col, color=color, alpha=0.7)
                    else:
                        ax_sub.plot(indicator_data.index, indicator_data[col], 
                                   label=col, color=color, linewidth=1.5)
            ax_sub.legend(loc='upper left', fontsize=10)
            ax_sub.grid(True, alpha=0.3)
            ax_sub.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        return fig

    def _plot_kline(self, ax: plt.Axes, data: pd.DataFrame):
        """
        在指定坐标轴上绘制K线

        Args:
            ax: matplotlib坐标轴
            data: OHLCV数据
        """
        dates = mdates.date2num(data.index)
        width = 0.6

        # 涨跌幅
        up = data[data['Close'] >= data['Open']]
        down = data[data['Close'] < data['Open']]

        # 绘制阳线
        ax.bar(mdates.date2num(up.index), up['Close'] - up['Open'], width,
               bottom=up['Open'], color=self.colors['red'], edgecolor=self.colors['red'])
        ax.bar(mdates.date2num(up.index), up['High'] - up['Close'], 0.1,
               bottom=up['Close'], color=self.colors['red'])
        ax.bar(mdates.date2num(up.index), up['Open'] - up['Low'], 0.1,
               bottom=up['Low'], color=self.colors['red'])

        # 绘制阴线
        ax.bar(mdates.date2num(down.index), down['Open'] - down['Close'], width,
               bottom=down['Close'], color=self.colors['green'], edgecolor=self.colors['green'])
        ax.bar(mdates.date2num(down.index), down['High'] - down['Open'], 0.1,
               bottom=down['Open'], color=self.colors['green'])
        ax.bar(mdates.date2num(down.index), down['Close'] - down['Low'], 0.1,
               bottom=down['Low'], color=self.colors['green'])

    def _plot_volume(self, ax: plt.Axes, data: pd.DataFrame):
        """
        在指定坐标轴上绘制成交量

        Args:
            ax: matplotlib坐标轴
            data: OHLCV数据
        """
        colors = np.where(data['Close'] >= data['Open'], self.colors['red'], self.colors['green'])
        ax.bar(data.index, data['Volume'], color=colors, alpha=0.7)
        ax.set_ylabel('成交量', fontsize=12)
        ax.grid(True, alpha=0.3)

    def plot_indicator_comparison(
        self,
        indicator_data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = '指标对比图',
        kind: str = 'line'
    ) -> plt.Figure:
        """
        绘制多个指标的对比图

        Args:
            indicator_data: 指标数据
            columns: 要展示的列名列表
            title: 图表标题
            kind: 图表类型 ('line', 'bar', 'area')

        Returns:
            matplotlib Figure对象
        """
        if columns is None:
            columns = indicator_data.columns.tolist()

        fig, ax = plt.subplots(figsize=self.figsize)

        color_list = list(self.colors.values())
        for i, col in enumerate(columns):
            if col in indicator_data.columns:
                color = color_list[i % len(color_list)]
                if kind == 'line':
                    ax.plot(indicator_data.index, indicator_data[col], 
                           label=col, color=color, linewidth=1.5, alpha=0.8)
                elif kind == 'bar':
                    ax.bar(indicator_data.index, indicator_data[col], 
                          label=col, color=color, alpha=0.7)
                elif kind == 'area':
                    ax.fill_between(indicator_data.index, indicator_data[col], 
                                   label=col, color=color, alpha=0.3)

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        return fig

    def plot_signal_scatter(
        self,
        data: pd.DataFrame,
        indicator_data: pd.DataFrame,
        signal_columns: List[str],
        title: str = '信号散点图'
    ) -> plt.Figure:
        """
        绘制买卖信号散点图

        Args:
            data: OHLCV数据
            indicator_data: 指标数据
            signal_columns: 信号列名列表（布尔值）
            title: 图表标题

        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # 绘制收盘价
        ax.plot(data.index, data['Close'], label='收盘价', 
               color=self.colors['gray'], linewidth=1, alpha=0.5)

        # 绘制信号点
        color_list = [self.colors['red'], self.colors['green'], self.colors['blue'], 
                     self.colors['yellow'], self.colors['purple']]
        marker_list = ['^', 'v', 'D', 's', 'o']

        for i, col in enumerate(signal_columns):
            if col in indicator_data.columns:
                signals = indicator_data[indicator_data[col] == True]
                if len(signals) > 0:
                    color = color_list[i % len(color_list)]
                    marker = marker_list[i % len(marker_list)]
                    ax.scatter(signals.index, data.loc[signals.index, 'Close'],
                              label=col, color=color, marker=marker, s=100, alpha=0.8, zorder=5)

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        return fig

    def plot_heatmap(
        self,
        indicator_data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = '指标热力图'
    ) -> plt.Figure:
        """
        绘制指标热力图

        Args:
            indicator_data: 指标数据
            columns: 要展示的列名列表
            title: 图表标题

        Returns:
            matplotlib Figure对象
        """
        if columns is None:
            columns = indicator_data.select_dtypes(include=[np.number]).columns.tolist()

        # 标准化数据
        normalized_data = indicator_data[columns].copy()
        for col in columns:
            normalized_data[col] = (indicator_data[col] - indicator_data[col].min()) / \
                                   (indicator_data[col].max() - indicator_data[col].min() + 1e-10)

        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(normalized_data.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')

        # 设置坐标轴标签
        ax.set_yticks(range(len(columns)))
        ax.set_yticklabels(columns, fontsize=10)
        
        # 设置X轴标签（每隔10个显示一个）
        tick_indices = range(0, len(normalized_data), max(1, len(normalized_data) // 10))
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([normalized_data.index[i].strftime('%Y-%m-%d') for i in tick_indices], 
                          rotation=45, fontsize=9)

        ax.set_title(title, fontsize=16, fontweight='bold')
        plt.colorbar(im, ax=ax, label='标准化值')

        plt.tight_layout()
        return fig

    def plot_composite_indicators(
        self,
        data: pd.DataFrame,
        banker_control_data: pd.DataFrame,
        main_capital_absorption_data: pd.DataFrame,
        main_cost_data: pd.DataFrame,
        main_trading_data: pd.DataFrame,
        title: str = '主力指标复合图',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制主力指标复合图

        Args:
            data: OHLCV数据
            banker_control_data: 主力控盘指标数据
            main_capital_absorption_data: 主力吸筹指标数据
            main_cost_data: 主力成本指标数据
            main_trading_data: 主力操盘指标数据
            title: 图表标题
            save_path: 保存路径（可选）

        Returns:
            matplotlib Figure对象
        """
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 1, 1, 1])

        # 主图：K线 + 主力操盘指标
        ax_main = fig.add_subplot(gs[0])
        self._plot_kline(ax_main, data)
        
        if 'attack_line' in main_trading_data.columns:
            ax_main.plot(main_trading_data.index, main_trading_data['attack_line'], 
                        label='攻击线', color=self.colors['yellow'], linewidth=1.5, alpha=0.8)
        if 'trading_line' in main_trading_data.columns:
            ax_main.plot(main_trading_data.index, main_trading_data['trading_line'], 
                        label='操盘线', color=self.colors['red'], linewidth=1.5, alpha=0.8)
        if 'defense_line' in main_trading_data.columns:
            ax_main.plot(main_trading_data.index, main_trading_data['defense_line'], 
                        label='防守线', color=self.colors['gray'], linewidth=1.5, alpha=0.8)
        
        if 'buy_signal' in main_trading_data.columns:
            buy_signals = main_trading_data[main_trading_data['buy_signal'] == 1]
            if len(buy_signals) > 0:
                ax_main.scatter(buy_signals.index, data.loc[buy_signals.index, 'Low'] * 0.99,
                               marker='^', color=self.colors['red'], s=100, zorder=5, label='买入信号')
        if 'sell_signal' in main_trading_data.columns:
            sell_signals = main_trading_data[main_trading_data['sell_signal'] == 1]
            if len(sell_signals) > 0:
                ax_main.scatter(sell_signals.index, data.loc[sell_signals.index, 'High'] * 1.01,
                               marker='v', color=self.colors['green'], s=100, zorder=5, label='卖出信号')
        
        ax_main.set_title(title, fontsize=16, fontweight='bold')
        ax_main.legend(loc='upper left', fontsize=10)
        ax_main.grid(True, alpha=0.3)
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45)

        # 子图1：主力控盘
        ax_control = fig.add_subplot(gs[1], sharex=ax_main)
        if 'control_degree' in banker_control_data.columns:
            control_degree = banker_control_data['control_degree']
            dates = mdates.date2num(banker_control_data.index)
            width = 0.6
            
            # 绘制不同区间的柱
            mask_50_60 = (control_degree >= 50) & (control_degree < 60)
            mask_60_80 = (control_degree >= 60) & (control_degree < 80)
            mask_80_plus = control_degree >= 80
            
            if mask_50_60.any():
                ax_control.bar(dates[mask_50_60], control_degree[mask_50_60], width, 
                              color=self.colors['yellow'], alpha=0.7)
            if mask_60_80.any():
                ax_control.bar(dates[mask_60_80], control_degree[mask_60_80], width, 
                              color=self.colors['red'], alpha=0.7)
            if mask_80_plus.any():
                ax_control.bar(dates[mask_80_plus], control_degree[mask_80_plus], width, 
                              color=self.colors['purple'], alpha=0.7)
            
            ax_control.axhline(y=50, color=self.colors['yellow'], linestyle='--', linewidth=0.5, alpha=0.5)
            ax_control.axhline(y=60, color=self.colors['red'], linestyle='--', linewidth=0.5, alpha=0.5)
            ax_control.axhline(y=80, color=self.colors['purple'], linestyle='--', linewidth=0.5, alpha=0.5)
        
        ax_control.set_ylabel('主力控盘', fontsize=12)
        ax_control.grid(True, alpha=0.3)

        # 子图2：主力吸筹
        ax_absorption = fig.add_subplot(gs[2], sharex=ax_main)
        if 'main_capital_absorption' in main_capital_absorption_data.columns:
            ax_absorption.bar(main_capital_absorption_data.index, 
                             main_capital_absorption_data['main_capital_absorption'],
                             facecolor='none', edgecolor=self.colors['purple'], linewidth=1.5, alpha=0.7)
        
        ax_absorption.set_ylabel('主力吸筹', fontsize=12)
        ax_absorption.grid(True, alpha=0.3)

        # 子图3：主力成本
        ax_cost = fig.add_subplot(gs[3], sharex=ax_main)
        if 'main_cost' in main_cost_data.columns:
            ax_cost.plot(main_cost_data.index, main_cost_data['main_cost'], 
                        label='主力成本', color=self.colors['yellow'], linewidth=2, alpha=0.8)
        
        ax_cost.set_ylabel('主力成本', fontsize=12)
        ax_cost.legend(loc='upper left', fontsize=10)
        ax_cost.grid(True, alpha=0.3)

        # 子图4：成交量
        ax_volume = fig.add_subplot(gs[4], sharex=ax_main)
        self._plot_volume(ax_volume, data)

        plt.tight_layout()

        if save_path:
            self.save_figure(fig, save_path)

        return fig

    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300):
        """
        保存图表到文件

        Args:
            fig: matplotlib Figure对象
            filename: 文件名
            dpi: 图片分辨率
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存至: {filename}")
        plt.close(fig)
