import logging
import numpy as np
import pandas as pd

from indicators.base import BaseIndicator

logger = logging.getLogger(__name__)


class MainCost(BaseIndicator):
    """
    主力成本指标

    该指标基于资金流向数据计算主力资金成本，支持真实资金流向数据和模拟数据两种模式。

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
    """

    def _simulate_capital_flow(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        模拟资金流向数据，当真实数据不可用时使用。

        Args:
            data: 包含OHLCV数据的输入DataFrame

        Returns:
            包含模拟资金流向数据的DataFrame
        """
        result = data.copy()

        np.random.seed(42)

        volume = data["Volume"]
        close = data["Close"]

        volume_ratio = np.random.uniform(0.1, 0.6, size=len(data))
        main_volume = volume * volume_ratio

        buy_ratio = np.random.uniform(0.3, 0.7, size=len(data))

        super_in = main_volume * buy_ratio * 0.4
        big_in = main_volume * buy_ratio * 0.6
        super_out = main_volume * (1 - buy_ratio) * 0.4
        big_out = main_volume * (1 - buy_ratio) * 0.6

        result["SYS_SUPERIN_TICK"] = super_in * close
        result["SYS_BIGIN_TICK"] = big_in * close
        result["SYS_SUPEROUT_TICK"] = super_out * close
        result["SYS_BIGOUT_TICK"] = big_out * close

        result["buy_count"] = np.random.randint(5, 50, size=len(data))
        result["sell_count"] = np.random.randint(5, 50, size=len(data))
        result["buy_total_price"] = (super_in + big_in) * close
        result["sell_total_price"] = (super_out + big_out) * close

        return result

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
            
            # 填充缺失值
            df['main_net_inflow'] = df['main_net_inflow'].fillna(0)
            df['big_net_inflow'] = df['big_net_inflow'].fillna(0)
            df['super_net_inflow'] = df['super_net_inflow'].fillna(0)
            df['small_net_inflow'] = df['small_net_inflow'].fillna(0)
            df['medium_net_inflow'] = df['medium_net_inflow'].fillna(0)
            
            # 使用真实资金流向数据计算
            df["main_buy"] = (df["super_net_inflow"] + df["big_net_inflow"]) / 10000
            df["main_sell"] = (-df["super_net_inflow"] - df["big_net_inflow"]) / 10000
            df["main_sell"] = df["main_sell"].clip(lower=0)
            
            df["net_buy"] = df["main_buy"] - df["main_sell"]
            df["cum_net_buy"] = df["net_buy"].cumsum()
            
            # 按策略公式计算主力成本
            df = self._calculate_by_strategy(df)
            
        else:
            # 使用模拟数据
            logger.warning("[主力成本指标] 未获取到真实资金流向数据，使用模拟数据替代！")
            df = self._simulate_capital_flow(data)

            df["main_buy"] = (df["SYS_SUPERIN_TICK"] + df["SYS_BIGIN_TICK"]) / 10000
            df["main_sell"] = (df["SYS_SUPEROUT_TICK"] + df["SYS_BIGOUT_TICK"]) / 10000
            df["net_buy"] = df["main_buy"] - df["main_sell"]
            df["cum_net_buy"] = df["net_buy"].cumsum()

            # 按策略公式计算主力成本
            df = self._calculate_by_strategy(df)

        result = data.copy()
        result["main_buy"] = df["main_buy"]
        result["main_sell"] = df["main_sell"]
        result["net_buy"] = df["net_buy"]
        result["cum_net_buy"] = df["cum_net_buy"]
        result["buy_avg_price"] = df["buy_avg_price"]
        result["sell_avg_price"] = df["sell_avg_price"]
        result["main_cost"] = df["main_cost"]
        result["avg_price"] = df["avg_price"]

        return result
    
    def _calculate_by_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按照策略公式计算主力成本。

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
            
            if pd.isna(close):
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
        df['avg_price'] = df['Close']
        
        return df
