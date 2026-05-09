import logging
import numpy as np
import pandas as pd

from indicators.base import BaseIndicator

logger = logging.getLogger(__name__)


class MainCost(BaseIndicator):
    """
    主力成本指标

    该指标基于真实资金流向数据计算主力资金成本。

    公式：
    - main_buy（万元）：(超大单净流入 + 大单净流入) / 10000
    - main_sell（万元）：(超大单净流出 + 大单净流出) / 10000
    - net_buy（万元）：main_buy - main_sell
    - cum_net_buy（万元）：SUM(net_buy, 0)
    - main_cost：基于资金流向计算的主力成本

    输出新增列：
    - main_buy：主力资金买入金额（万元）
    - main_sell：主力资金卖出金额（万元）
    - net_buy：净买入金额（万元）
    - cum_net_buy：累计净买入金额（万元）
    - main_cost：主力资金成本价
    - main_net_buy_wan：主力净买入（万元）
    - main_direction：主力方向（inflow/outflow）
    - turnover_ratio：主力成交占比（%）
    """



    def calculate(self, data: pd.DataFrame, fund_flow_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        计算主力成本指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame（Open, High, Low, Close, Volume）
            fund_flow_data: 包含资金流向数据的DataFrame（可选）

        Returns:
            添加了'main_buy', 'main_sell', 'net_buy', 'cum_net_buy', 
            'buy_avg_price', 'sell_avg_price', 'main_cost', 'avg_price'列的DataFrame
        """
        self.validate_input(data)

        if fund_flow_data is not None and not fund_flow_data.empty:
            logger.info("[主力成本指标] 使用真实资金流向数据计算主力成本")
            df = data.copy()
            
            # 确保日期格式一致
            if 'date' in fund_flow_data.columns:
                fund_flow_data['date'] = pd.to_datetime(fund_flow_data['date']).dt.date
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
            
            # 按日期合并数据
            df = df.merge(
                fund_flow_data[['date', 'main_net_inflow', 'big_net_inflow', 'super_net_inflow', 
                               'small_net_inflow', 'medium_net_inflow']],
                on='date',
                how='left'
            )
            
            # 只在有真实资金流向数据的日期计算，其他日期保持NaN
            mask = df['main_net_inflow'].notna()
            if mask.any():
                logger.info(f"[主力成本指标] 使用 {mask.sum()} 天真实资金流向数据")
                
                # 优先使用原始主力净流入数据，降级使用超大单+大单计算
                # 这样既保证了数据准确性，又增强了容错性
                if "main_net_inflow" in df.columns and df["main_net_inflow"].notna().any():
                    logger.info("[主力成本指标] 使用原始主力净流入数据")
                    df.loc[mask, "main_buy"] = np.where(df.loc[mask, "main_net_inflow"] > 0, df.loc[mask, "main_net_inflow"] / 10000, 0)
                    df.loc[mask, "main_sell"] = np.where(df.loc[mask, "main_net_inflow"] < 0, -df.loc[mask, "main_net_inflow"] / 10000, 0)
                else:
                    logger.warning("[主力成本指标] 原始主力净流入数据不可用，降级使用超大单+大单计算")
                    df.loc[mask, "main_buy"] = (df.loc[mask, "super_net_inflow"] + df.loc[mask, "big_net_inflow"]) / 10000
                    df.loc[mask, "main_sell"] = (-df.loc[mask, "super_net_inflow"] - df.loc[mask, "big_net_inflow"]) / 10000
                    df.loc[mask, "main_sell"] = df.loc[mask, "main_sell"].clip(lower=0)
                
                df.loc[mask, "net_buy"] = df.loc[mask, "main_buy"] - df.loc[mask, "main_sell"]
                df.loc[mask, "cum_net_buy"] = df.loc[mask, "net_buy"].cumsum()
                
                # 追加主力资金相关数据，供前端十字线使用
                # 优先用原始 main_net_inflow，保证数据准确
                df.loc[mask, "main_net_buy_wan"] = df.loc[mask, "main_net_inflow"] / 10000
                
                # 更高效的向量化方向计算 - 用原始数据判断
                df["main_direction"] = None
                df.loc[mask & (df["main_net_inflow"] > 0), "main_direction"] = "inflow"
                df.loc[mask & (df["main_net_inflow"] < 0), "main_direction"] = "outflow"
                
                # 计算成交占比 - 优先用东方财富现成的 main_net_ratio，没有的话自己算
                if "main_net_ratio" in df.columns:
                    df.loc[mask, "turnover_ratio"] = df.loc[mask, "main_net_ratio"].round(2)
                else:
                    amount_col = "amount" if "amount" in df.columns else "Amount" if "Amount" in df.columns else None
                    if amount_col:
                        df.loc[mask, "turnover_ratio"] = (
                            (df.loc[mask, "main_net_inflow"].abs() / df.loc[mask, amount_col] * 100)
                            .round(2)
                        )
            
            # 按策略公式计算主力成本
            df = self._calculate_by_strategy(df)
            
        else:
            logger.warning("[主力成本指标] 未获取到真实资金流向数据，无法计算主力成本！")
            result = data.copy()
            result["main_buy"] = np.nan
            result["main_sell"] = np.nan
            result["net_buy"] = np.nan
            result["cum_net_buy"] = np.nan
            result["buy_avg_price"] = np.nan
            result["sell_avg_price"] = np.nan
            result["main_cost"] = np.nan
            result["avg_price"] = np.nan
            return result

        # 直接从 df 复制所有数据（包含所有原始 K 线列和我们新增的指标列）
        result = df.copy()
        return result
    
    def _calculate_by_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按照策略公式计算主力成本。只在有真实资金流向数据的日期计算，其他日期保持NaN。

        Args:
            df: 包含数据的DataFrame

        Returns:
            添加了计算结果的DataFrame
        """
        # 初始化累计变量
        buy_total = 0.0  # 买总价
        buy_count = 0  # 买次数
        sell_total = 0.0  # 卖总价
        sell_count = 0  # 卖次数
        
        buy_avg_series = []
        sell_avg_series = []
        main_cost_series = []
        
        close_prices = df['Close'].values
        main_buy_values = df['main_buy'].values
        main_sell_values = df['main_sell'].values
        
        for i in range(len(df)):
            close = close_prices[i]
            main_buy = main_buy_values[i]
            main_sell = main_sell_values[i]
            
            # 只在有真实资金流向数据的日期计算
            if pd.isna(main_buy) or pd.isna(main_sell) or pd.isna(close):
                buy_avg_series.append(np.nan)
                sell_avg_series.append(np.nan)
                main_cost_series.append(np.nan)
                continue
            
            # BGJ: IF(主力买入万元>0, CLOSE, DRAWNULL)
            if main_buy > 0:
                buy_total += close
                buy_count += 1
            
            # SGJ: IF(主力卖出万元>0, CLOSE, DRAWNULL)
            if main_sell > 0:
                sell_total += close
                sell_count += 1
            
            # 买均价
            buy_avg = buy_total / buy_count if buy_count > 0 else np.nan
            buy_avg_series.append(buy_avg)
            
            # 卖均价
            sell_avg = sell_total / sell_count if sell_count > 0 else np.nan
            sell_avg_series.append(sell_avg)
            
            # 主力成本
            total_price = buy_total + sell_total
            total_count = buy_count + sell_count
            main_cost = total_price / total_count if total_count > 0 else np.nan
            main_cost_series.append(main_cost)
        
        df['buy_avg_price'] = buy_avg_series
        df['sell_avg_price'] = sell_avg_series
        df['main_cost'] = main_cost_series
        
        # 成交均价线：DYNAINFO(11) - 使用平均成交价（这里简化为Close，因为没有单独的成交价数据）
        df['avg_price'] = np.where(pd.notna(df['main_cost']), df['Close'], np.nan)
        
        return df
