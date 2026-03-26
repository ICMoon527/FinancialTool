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

    def calculate(self, data: pd.DataFrame, fund_flow_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate the Main Capital Flow indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume
            fund_flow_data: 包含资金流向数据的DataFrame（可选）

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        if fund_flow_data is not None and not fund_flow_data.empty:
            # 使用真实资金流向数据
            # 确保日期格式一致
            if 'date' in fund_flow_data.columns:
                fund_flow_data['date'] = pd.to_datetime(fund_flow_data['date']).dt.date
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
            
            # 按日期合并数据
            df = df.merge(
                fund_flow_data[['date', 'main_net_inflow', 'big_net_inflow', 'super_net_inflow']],
                on='date',
                how='left'
            )
            
            # 填充缺失值
            df['main_net_inflow'] = df['main_net_inflow'].fillna(0)
            df['big_net_inflow'] = df['big_net_inflow'].fillna(0)
            df['super_net_inflow'] = df['super_net_inflow'].fillna(0)
            
            # 使用真实资金流向数据计算
            main_net_buy = (df['super_net_inflow'] + df['big_net_inflow']) / 10000
            result['main_net_buy_wan'] = main_net_buy
            
            # 分离买入和卖出
            result['main_buy_wan'] = main_net_buy.where(main_net_buy > 0, 0)
            result['main_sell_wan'] = -main_net_buy.where(main_net_buy < 0, 0)
            
        else:
            # 使用模拟数据（价格和成交量作为代理）
            price_change = df['Close'] - df['Open']
            
            # 主力买入万元: 价格上涨时的成交量
            main_buy = df['Volume'] * df['Close'] / 10000
            main_buy = main_buy.where(price_change > 0, 0)
            result['main_buy_wan'] = main_buy

            # 主力卖出万元: 价格下跌时的成交量
            main_sell = df['Volume'] * df['Close'] / 10000
            main_sell = main_sell.where(price_change < 0, 0)
            result['main_sell_wan'] = main_sell

            # 主力净买万元: 主力买入万元-主力卖出万元
            main_net_buy = main_buy - main_sell
            result['main_net_buy_wan'] = main_net_buy

        # 成交占比: 100*(主力净买万元*10000)/AMOUNT
        if 'Amount' in df.columns:
            turnover_ratio = 100 * (result['main_net_buy_wan'] * 10000) / (df['Amount'] + 1e-10)
        else:
            turnover_ratio = 100 * result['main_net_buy_wan'] / (df['Volume'] * df['Close'] / 10000 + 1e-10)
        result['turnover_ratio'] = turnover_ratio

        # 红柱: 主力净买万元>0
        result['red_bar'] = result['main_net_buy_wan'] > 0

        # 绿柱: 主力净买万元<0
        result['green_bar'] = result['main_net_buy_wan'] < 0

        return result
