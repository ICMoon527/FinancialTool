import pandas as pd
import numpy as np
from ..base import BaseIndicator


class CaptureDragonTracking(BaseIndicator):
    """
    擒龙追踪指标
    红白细线代表主力攻击线，黄粗线代表主力控盘线
    红色主力攻击线表示主力攻击线在主力控盘线之上且主力攻击线向上
    紫色3D效果K线表示股价连续涨停
    火箭图标表示龙头启动信号、星星图标表示龙头主升信号
    钻石信号表示龙头二波信号，旗子图标表示创业板和科创板个股强趋势信号
    """

    def __init__(self):
        super().__init__()

    def _sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def _hhv(self, data: pd.Series, period: int) -> pd.Series:
        """Highest High Value"""
        return data.rolling(window=period).max()

    def _llv(self, data: pd.Series, period: int) -> pd.Series:
        """Lowest Low Value"""
        return data.rolling(window=period).min()

    def _count(self, condition: pd.Series, period: int) -> pd.Series:
        """Count of condition being true over period"""
        return condition.astype(int).rolling(window=period).sum()

    def _barslast(self, condition: pd.Series) -> pd.Series:
        """Bars since last occurrence of condition"""
        result = pd.Series(np.nan, index=condition.index)
        last_bar = -1
        for i in range(len(condition)):
            if condition.iloc[i]:
                last_bar = i
                result.iloc[i] = 0
            elif last_bar != -1:
                result.iloc[i] = i - last_bar
        return result

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Capture Dragon Tracking indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)

        df = data.copy()
        result = pd.DataFrame(index=df.index)

        # 主力攻击线: (MA(CLOSE,3)+MA(CLOSE,6)+MA(CLOSE,9))/3
        ma3 = self._sma(df['Close'], 3)
        ma6 = self._sma(df['Close'], 6)
        ma9 = self._sma(df['Close'], 9)
        result['main_attack_line'] = (ma3 + ma6 + ma9) / 3

        # 主力控盘线: (MA(CLOSE,5)+MA(CLOSE,10)+MA(CLOSE,15)+MA(CLOSE,20)+MA(CLOSE,25)+MA(CLOSE,30))/6
        ma5 = self._sma(df['Close'], 5)
        ma10 = self._sma(df['Close'], 10)
        ma15 = self._sma(df['Close'], 15)
        ma20 = self._sma(df['Close'], 20)
        ma25 = self._sma(df['Close'], 25)
        ma30 = self._sma(df['Close'], 30)
        result['main_control_line'] = (ma5 + ma10 + ma15 + ma20 + ma25 + ma30) / 6

        # 红色主力攻击线条件
        result['attack_line_red'] = (
            (result['main_attack_line'] > result['main_attack_line'].shift(1)) &
            (result['main_attack_line'] > result['main_control_line'])
        )

        # 涨停: 涨幅>=0.09 AND CLOSE=HIGH
        change_pct = df['Close'] / df['Close'].shift(1) - 1
        limit_up = (change_pct >= 0.09) & (df['Close'] == df['High'])
        result['limit_up'] = limit_up

        # 实体: CLOSE>LOW
        result['has_body'] = df['Close'] > df['Low']

        # 成交额: AMOUNT>=REF(HHV(AMOUNT,10),1)
        if 'Amount' in df.columns:
            hhv_amount_10 = self._hhv(df['Amount'], 10)
            result['turnover_condition'] = df['Amount'] >= hhv_amount_10.shift(1)
        else:
            result['turnover_condition'] = True

        # RSV: (CLOSE-LLV(LOW,34))/(HHV(HIGH,34)-LLV(LOW,34))*100
        llv_low_34 = self._llv(df['Low'], 34)
        hhv_high_34 = self._hhv(df['High'], 34)
        rsv = ((df['Close'] - llv_low_34) / (hhv_high_34 - llv_low_34 + 1e-10)) * 100

        # K:=SMA(RSV,3,1)
        k = self._sma(rsv, 3)
        result['k'] = k

        # 乖离率: (CLOSE/主力攻击线-1)*100
        result['bias_rate'] = (df['Close'] / result['main_attack_line'] - 1) * 100

        # 连板: COUNT(涨停,2)=2
        consecutive_limit_up = self._count(limit_up, 2) == 2
        result['consecutive_limit_up'] = consecutive_limit_up

        # A2ZQ:=BARSLAST(连板)
        a2zq = self._barslast(consecutive_limit_up)
        result['a2zq'] = a2zq

        # 连续涨停K线
        result['continuous_limit_up_candle'] = (a2zq == 0)

        # 均线金叉: BARSLAST(CROSS(主力攻击线,主力控盘线))<=8 AND 乖离率<=10 AND 主力攻击线>主力控盘线 AND C>主力攻击线
        golden_cross = pd.Series(False, index=df.index)
        attack_above_control = result['main_attack_line'] > result['main_control_line']
        attack_cross_control = (attack_above_control & ~attack_above_control.shift(1).fillna(False).astype(bool))
        bars_since_cross = self._barslast(attack_cross_control)
        
        golden_cross = (
            (bars_since_cross <= 8) &
            (result['bias_rate'] <= 10) &
            attack_above_control &
            (df['Close'] > result['main_attack_line'])
        )
        result['golden_cross'] = golden_cross

        # 持续放量: 10000*SUM(VOL,3)/FINANCE(7)>=10 AND COUNT(AMOUNT>REF(AMOUNT,1),3)=3
        # Note: FINANCE(7) is not available, using simplified proxy
        if 'Amount' in df.columns:
            amount_increasing = self._count(df['Amount'] > df['Amount'].shift(1), 3) == 3
            result['sustained_volume'] = amount_increasing
        else:
            result['sustained_volume'] = True

        # 超跌板: 涨停=1 AND 实体=1 AND COUNT(涨停,13)=1 AND REF(K,1)<=20 AND 成交额=1
        super_drop_limit_up = (
            limit_up &
            result['has_body'] &
            (self._count(limit_up, 13) == 1) &
            (k.shift(1) <= 20) &
            result['turnover_condition']
        )
        result['super_drop_limit_up'] = super_drop_limit_up

        # 趋势板: 涨停=1 AND 实体=1 AND COUNT(涨停,13)=1 AND 均线金叉=1 AND 成交额=1
        trend_limit_up = (
            limit_up &
            result['has_body'] &
            (self._count(limit_up, 13) == 1) &
            golden_cross &
            result['turnover_condition']
        )
        result['trend_limit_up'] = trend_limit_up

        # 放量板: 涨停=1 AND 实体=1 AND COUNT(涨停,13)=1 AND 持续放量=1 AND 成交额=1
        volume_limit_up = (
            limit_up &
            result['has_body'] &
            (self._count(limit_up, 13) == 1) &
            result['sustained_volume'] &
            result['turnover_condition']
        )
        result['volume_limit_up'] = volume_limit_up

        # XG:=放量板 OR 趋势板 OR 超跌板
        result['xg'] = volume_limit_up | trend_limit_up | super_drop_limit_up

        # 信号汇总
        result['super_drop_signal'] = super_drop_limit_up
        result['trend_signal'] = trend_limit_up
        result['volume_signal'] = volume_limit_up

        return result
