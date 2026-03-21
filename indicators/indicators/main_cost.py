import numpy as np
import pandas as pd

from indicators.base import BaseIndicator


class MainCost(BaseIndicator):
    """
    主力成本指标

    该指标基于资金流向数据计算主力资金成本，模拟超大单和大单交易，并计算主力资金的平均价格和累计净买入。

    公式：
    - main_buy（万元）：(SYS_SUPERIN_TICK + SYS_BIGIN_TICK) / 10000
    - main_sell（万元）：(SYS_SUPEROUT_TICK + SYS_BIGOUT_TICK) / 10000
    - net_buy（万元）：main_buy - main_sell
    - cum_net_buy（万元）：SUM(net_buy, 0)
    - buy_avg_price：buy_total_price / buy_count
    - sell_avg_price：sell_total_price / sell_count
    - main_cost：(sell_total_price + buy_total_price) / (sell_count + buy_count)

    输出新增列：
    - main_buy：主力资金买入金额（万元）
    - main_sell：主力资金卖出金额（万元）
    - net_buy：净买入金额（万元）
    - cum_net_buy：累计净买入金额（万元）
    - buy_avg_price：平均买入价格
    - sell_avg_price：平均卖出价格
    - main_cost：主力资金成本价
    """

    def _simulate_capital_flow(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        模拟资金流向数据，因为真实数据可能不可用。

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

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算主力成本指标。

        Args:
            data: 包含OHLCV数据的输入DataFrame（Open, High, Low, Close, Volume）

        Returns:
            添加了'main_buy', 'main_sell', 'net_buy', 'cum_net_buy',
            'buy_avg_price', 'sell_avg_price', 'main_cost'列的DataFrame
        """
        self.validate_input(data)

        df = self._simulate_capital_flow(data)

        df["main_buy"] = (df["SYS_SUPERIN_TICK"] + df["SYS_BIGIN_TICK"]) / 10000
        df["main_sell"] = (df["SYS_SUPEROUT_TICK"] + df["SYS_BIGOUT_TICK"]) / 10000
        df["net_buy"] = df["main_buy"] - df["main_sell"]
        df["cum_net_buy"] = df["net_buy"].cumsum()

        df["buy_avg_price"] = df["buy_total_price"] / df["buy_count"].replace(0, np.nan)
        df["sell_avg_price"] = df["sell_total_price"] / df["sell_count"].replace(0, np.nan)

        total_price = df["buy_total_price"] + df["sell_total_price"]
        total_count = df["buy_count"] + df["sell_count"]
        df["main_cost"] = total_price / total_count.replace(0, np.nan)

        result = data.copy()
        result["main_buy"] = df["main_buy"]
        result["main_sell"] = df["main_sell"]
        result["net_buy"] = df["net_buy"]
        result["cum_net_buy"] = df["cum_net_buy"]
        result["buy_avg_price"] = df["buy_avg_price"]
        result["sell_avg_price"] = df["sell_avg_price"]
        result["main_cost"] = df["main_cost"]

        return result
