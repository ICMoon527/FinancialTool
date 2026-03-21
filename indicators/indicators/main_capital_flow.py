import pandas as pd
import numpy as np
from ..base import BaseIndicator


class MainCapitalFlow(BaseIndicator):
    """
    主力流向指标
    红柱表示当日主力资金净流入
    绿柱表示当日主力资金净流出
    """

    def __init__(self):
        super().__init__()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Main Capital Flow indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        # 主力买入万元0:=(SYS_SUPERIN+SYS_BIGIN)/10000
        # 使用价格和成交量作为代理
        price_change = df['Close'] - df['Open']
        
        # 主力买入万元: 价格上涨时的成交量
        main_buy = df['Volume'] * df['Close'] / 10000
        main_buy = main_buy.where(price_change > 0, 0)
        result['main_buy_wan'] = main_buy

        # 主力卖出万元0:=(SYS_SUPEROUT+SYS_BIGOUT)/10000
        # 主力卖出万元: 价格下跌时的成交量
        main_sell = df['Volume'] * df['Close'] / 10000
        main_sell = main_sell.where(price_change < 0, 0)
        result['main_sell_wan'] = main_sell

        # 主力净买万元: 主力买入万元0-主力卖出万元0
        main_net_buy = main_buy - main_sell
        result['main_net_buy_wan'] = main_net_buy

        # 成交占比: 100*(主力净买万元*10000)/AMOUNT
        if 'Amount' in df.columns:
            turnover_ratio = 100 * (main_net_buy * 10000) / (df['Amount'] + 1e-10)
        else:
            turnover_ratio = 100 * main_net_buy / (df['Volume'] * df['Close'] / 10000 + 1e-10)
        result['turnover_ratio'] = turnover_ratio

        # 红柱: 主力净买万元>0
        result['red_bar'] = main_net_buy > 0

        # 绿柱: 主力净买万元<0
        result['green_bar'] = main_net_buy < 0

        return result
