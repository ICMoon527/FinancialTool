# -*- coding: utf-8 -*-
"""
回测报告生成模块。

生成包含策略逻辑、数据来源、回测设置、指标计算方法及结果分析的完整报告。
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .engine import Portfolio
from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class BacktestReportGenerator:
    """
    回测报告生成器。
    """

    def __init__(
        self,
        portfolio: Portfolio,
        metrics: PerformanceMetrics,
        strategy_name: str = "策略",
        output_dir: str = "backtest_results",
    ):
        """
        初始化报告生成器。

        Args:
            portfolio: 投资组合
            metrics: 绩效指标
            strategy_name: 策略名称
            output_dir: 输出目录
        """
        self.portfolio = portfolio
        self.metrics = metrics
        self.strategy_name = strategy_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(
        self,
        backtest_config: Optional[Dict[str, Any]] = None,
        chart_paths: Optional[Dict[str, str]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """
        生成Markdown格式的回测报告。

        Args:
            backtest_config: 回测配置
            chart_paths: 图表路径字典
            save_path: 保存路径

        Returns:
            报告文件路径
        """
        all_metrics = self.metrics.get_all_metrics()

        report_content = []

        report_content.append(f"# {self.strategy_name} - 回测报告\n")
        report_content.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_content.append("\n---\n")

        report_content.append("## 1. 策略概述\n")
        report_content.append(f"- **策略名称**: {self.strategy_name}\n")
        report_content.append("- **策略类型**: 六维选股策略\n")
        report_content.append("- **策略说明**: 整合主力操盘、庄家控盘、动能二号、共振追涨、强势起爆五个策略的筛选逻辑。\n")
        report_content.append("\n")

        report_content.append("## 2. 回测设置\n")
        if backtest_config:
            for key, value in backtest_config.items():
                report_content.append(f"- **{key}**: {value}\n")
        report_content.append(f"- **初始资金**: {self.portfolio.initial_capital:,.2f} 元\n")
        report_content.append("\n")

        report_content.append("## 3. 数据来源\n")
        report_content.append("- **行情数据**: AkShare / Tushare / Baostock\n")
        report_content.append("- **数据周期**: 日线数据\n")
        report_content.append("\n")

        report_content.append("## 4. 交易规则\n")
        report_content.append("- **调仓频率**: 日度调仓\n")
        report_content.append("- **交易成本**: 包含手续费和滑点\n")
        report_content.append("- **下单方式**: 市价单\n")
        report_content.append("\n")

        report_content.append("## 5. 绩效指标\n")
        report_content.append("\n### 5.1 收益指标\n")
        report_content.append(f"- **总收益率**: {all_metrics.get('total_return', 0)*100:.2f}%\n")
        report_content.append(f"- **年化收益率**: {all_metrics.get('annualized_return', 0)*100:.2f}%\n")
        report_content.append("\n### 5.2 风险指标\n")
        report_content.append(f"- **年化波动率**: {all_metrics.get('volatility', 0)*100:.2f}%\n")
        report_content.append(f"- **最大回撤**: {all_metrics.get('max_drawdown', 0)*100:.2f}%\n")
        if all_metrics.get('max_drawdown_start'):
            report_content.append(f"  - 回撤开始: {all_metrics.get('max_drawdown_start')}\n")
        if all_metrics.get('max_drawdown_end'):
            report_content.append(f"  - 回撤结束: {all_metrics.get('max_drawdown_end')}\n")
        report_content.append("\n### 5.3 风险调整收益指标\n")
        report_content.append(f"- **夏普比率**: {all_metrics.get('sharpe_ratio', 0):.4f}\n")
        report_content.append(f"- **卡尔马比率**: {all_metrics.get('calmar_ratio', 0):.4f}\n")
        report_content.append(f"- **索提诺比率**: {all_metrics.get('sortino_ratio', 0):.4f}\n")
        report_content.append(f"- **信息比率**: {all_metrics.get('information_ratio', 0):.4f}\n")
        report_content.append("\n### 5.4 阿尔法与贝塔\n")
        report_content.append(f"- **贝塔 (Beta)**: {all_metrics.get('beta', 0):.4f}\n")
        report_content.append(f"- **阿尔法 (Alpha)**: {all_metrics.get('alpha', 0)*100:.4f}%\n")
        report_content.append("\n### 5.5 交易指标\n")
        report_content.append(f"- **总交易次数**: {all_metrics.get('total_trades', 0)}\n")
        report_content.append(f"- **胜率**: {all_metrics.get('win_rate', 0)*100:.2f}%\n")
        report_content.append(f"- **盈亏比**: {all_metrics.get('profit_loss_ratio', 0):.4f}\n")
        report_content.append("\n### 5.6 最终结果\n")
        report_content.append(f"- **初始资金**: {all_metrics.get('initial_capital', 0):,.2f} 元\n")
        report_content.append(f"- **最终权益**: {all_metrics.get('final_equity', 0):,.2f} 元\n")
        report_content.append("\n")

        report_content.append("## 6. 指标计算方法\n")
        report_content.append("\n### 6.1 年化收益率\n")
        report_content.append("```\n年化收益率 = (1 + 总收益率) ^ (1 / 年数) - 1\n```\n")
        report_content.append("\n### 6.2 夏普比率\n")
        report_content.append("```\n夏普比率 = (年化收益率 - 无风险利率) / 年化波动率\n```\n")
        report_content.append("\n### 6.3 最大回撤\n")
        report_content.append("```\n最大回撤 = max( (历史最高值 - 当前值) / 历史最高值 )\n```\n")
        report_content.append("\n### 6.4 卡尔马比率\n")
        report_content.append("```\n卡尔马比率 = 年化收益率 / 最大回撤\n```\n")
        report_content.append("\n### 6.5 胜率\n")
        report_content.append("```\n胜率 = 盈利交易次数 / 总交易次数\n```\n")
        report_content.append("\n### 6.6 盈亏比\n")
        report_content.append("```\n盈亏比 = 平均盈利 / 平均亏损\n```\n")
        report_content.append("\n### 6.7 贝塔 (Beta)\n")
        report_content.append("```\nBeta = Cov(投资组合收益率, 基准收益率) / Var(基准收益率)\n```\n")
        report_content.append("- Beta = 1：投资组合的风险与基准相同\n")
        report_content.append("- Beta > 1：投资组合的风险高于基准（激进型）\n")
        report_content.append("- Beta < 1：投资组合的风险低于基准（保守型）\n")
        report_content.append("\n### 6.8 阿尔法 (Alpha)\n")
        report_content.append("```\nAlpha = 投资组合年化收益率 - [无风险利率 + Beta * (基准年化收益率 - 无风险利率)]\n```\n")
        report_content.append("- Alpha > 0：投资组合表现优于基准\n")
        report_content.append("- Alpha = 0：投资组合表现与基准相同\n")
        report_content.append("- Alpha < 0：投资组合表现不如基准\n")
        report_content.append("\n")

        if chart_paths:
            report_content.append("## 7. 可视化图表\n")
            for chart_name, chart_path in chart_paths.items():
                chart_file = Path(chart_path).name
                report_content.append(f"\n### 7.{list(chart_paths.keys()).index(chart_name)+1} {chart_name.replace('_', ' ').title()}\n")
                report_content.append(f"![{chart_name}]({chart_file})\n")
            report_content.append("\n")

        report_content.append("## 8. 结果分析\n")
        report_content.append("\n### 8.1 收益表现\n")
        total_return = all_metrics.get('total_return', 0)
        if total_return > 0:
            report_content.append(f"- 策略在回测期间取得了正收益，总收益率为 {total_return*100:.2f}%\n")
        else:
            report_content.append(f"- 策略在回测期间收益为负，总收益率为 {total_return*100:.2f}%\n")
        report_content.append("\n### 8.2 风险控制\n")
        max_drawdown = all_metrics.get('max_drawdown', 0)
        if max_drawdown < 0.2:
            report_content.append(f"- 最大回撤为 {max_drawdown*100:.2f}%，风险控制较好\n")
        else:
            report_content.append(f"- 最大回撤为 {max_drawdown*100:.2f}%，需要注意风险控制\n")
        report_content.append("\n### 8.3 风险调整收益\n")
        sharpe_ratio = all_metrics.get('sharpe_ratio', 0)
        if sharpe_ratio > 1:
            report_content.append(f"- 夏普比率为 {sharpe_ratio:.4f}，风险调整收益优秀\n")
        elif sharpe_ratio > 0:
            report_content.append(f"- 夏普比率为 {sharpe_ratio:.4f}，风险调整收益尚可\n")
        else:
            report_content.append(f"- 夏普比率为 {sharpe_ratio:.4f}，风险调整收益不佳\n")
        report_content.append("\n### 8.4 交易表现\n")
        win_rate = all_metrics.get('win_rate', 0)
        profit_loss_ratio = all_metrics.get('profit_loss_ratio', 0)
        if win_rate > 0.5 and profit_loss_ratio > 1:
            report_content.append(f"- 胜率 {win_rate*100:.2f}%，盈亏比 {profit_loss_ratio:.4f}，交易表现优秀\n")
        else:
            report_content.append(f"- 胜率 {win_rate*100:.2f}%，盈亏比 {profit_loss_ratio:.4f}\n")
        report_content.append("\n### 8.5 阿尔法与贝塔分析\n")
        beta = all_metrics.get('beta', 0)
        alpha = all_metrics.get('alpha', 0)
        if beta > 1.2:
            report_content.append(f"- 贝塔值为 {beta:.4f}，策略表现相对激进，波动大于市场\n")
        elif beta > 0.8:
            report_content.append(f"- 贝塔值为 {beta:.4f}，策略风险与市场接近\n")
        else:
            report_content.append(f"- 贝塔值为 {beta:.4f}，策略表现相对保守，波动小于市场\n")
        if alpha > 0:
            report_content.append(f"- 阿尔法值为 {alpha*100:.4f}%，策略表现优于基准\n")
        elif alpha == 0:
            report_content.append(f"- 阿尔法值为 {alpha*100:.4f}%，策略表现与基准持平\n")
        else:
            report_content.append(f"- 阿尔法值为 {alpha*100:.4f}%，策略表现不如基准\n")
        report_content.append("\n")

        report_content.append("## 9. 免责声明\n")
        report_content.append("- 本回测结果仅供参考，不构成投资建议\n")
        report_content.append("- 历史表现不代表未来收益\n")
        report_content.append("- 市场有风险，投资需谨慎\n")
        report_content.append("\n")

        report_content.append("---\n")
        report_content.append(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        if save_path is None:
            save_path = str(self.output_dir / "backtest_report.md")

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_content))

        logger.info(f"回测报告已保存至: {save_path}")
        return save_path
