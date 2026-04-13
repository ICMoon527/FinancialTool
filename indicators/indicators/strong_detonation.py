import logging
import pandas as pd
import numpy as np
from ..base import BaseIndicator

logger = logging.getLogger(__name__)


class StrongDetonation(BaseIndicator):
    """
    强势起爆指标
    黄色线是个股强势趋势线，K线高于黄色线表示个股趋势走强
    红色网状线是大盘走势，K线高于红色网状线表示个股中期跑赢大市
    紫色K线表示个股同时高于黄色线和红色网状线，股价处于强势起爆阶段
    红色K线表示个股高于红色网状线低于黄色线，股价处于强势蓄势阶段
    灰色K线表示个股处于弱势震荡阶段
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

    def calculate(self, data: pd.DataFrame, index_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate the Strong Detonation indicator

        Args:
            data: Input DataFrame with columns: Open, High, Low, Close, Volume
            index_data: 包含大盘指数历史数据的DataFrame（可选）

        Returns:
            DataFrame with indicator values
        """
        self.validate_input(data)
        logger.info("[强势起爆指标] 数据验证通过，开始计算指标")

        df = data.copy()
        result = df.copy()
        
        # 大盘中线计算
        ema120_stock = None
        market_midline = None
        
        if index_data is not None and not index_data.empty:
            logger.info("[强势起爆指标] 使用真实大盘指数数据计算大盘中线")
            
            try:
                # 确保日期格式一致
                if 'date' in index_data.columns:
                    index_data['date'] = pd.to_datetime(index_data['date']).dt.date
                
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.date
                
                # 先重命名大盘数据的close列，避免与个股数据的Close列冲突
                index_data_renamed = index_data[['date', 'close']].rename(columns={'close': 'close_index'})
                
                # 按日期合并数据
                df = df.merge(
                    index_data_renamed,
                    on='date',
                    how='left'
                )
                
                # 检查是否成功合并
                if 'close_index' in df.columns:
                    # 填充缺失值，使用前向填充和后向填充
                    df['close_index'] = df['close_index'].ffill().bfill()
                    
                    # 从 merged_df 统一获取所有数据，确保索引一致
                    close = df['Close']
                    open_price = df['Open']
                    high = df['High']
                    low = df['Low']
                    
                    # AAA:=(3*C+H+L+O)/6
                    aaa = (3 * close + high + low + open_price) / 6
                    result['aaa'] = aaa

                    # VAR1:=EMA(AAA,35)
                    var1 = self._ema(aaa, 35)
                    result['var1'] = var1

                    # VAR2:=(HHV(VAR1,5)+HHV(VAR1,15)+HHV(VAR1,30))/3
                    var2 = (self._hhv(var1, 5) + self._hhv(var1, 15) + self._hhv(var1, 30)) / 3
                    result['var2'] = var2

                    # VAR3:=(LLV(VAR1,5)+LLV(VAR1,15)+LLV(VAR1,30))/3
                    var3 = (self._llv(var1, 5) + self._llv(var1, 15) + self._llv(var1, 30)) / 3
                    result['var3'] = var3

                    # 牛线:=(HHV(VAR2,5)+HHV(VAR2,15)+HHV(VAR2,30))/3
                    bull_line = (self._hhv(var2, 5) + self._hhv(var2, 15) + self._hhv(var2, 30)) / 3
                    result['bull_line'] = bull_line

                    # 熊线:=(LLV(VAR3,5)+LLV(VAR3,15)+LLV(VAR3,30))/3
                    bear_line = (self._llv(var3, 5) + self._llv(var3, 15) + self._llv(var3, 30)) / 3
                    result['bear_line'] = bear_line

                    # 计算个股的长期趋势（EMA120）
                    ema120_stock = self._ema(close, 120)
                    
                    # 检查是否还有缺失值
                    if df['close_index'].isna().any():
                        logger.warning("[强势起爆指标] 部分大盘数据缺失，使用个股价格作为代理")
                        # 使用个股价格作为代理
                        a1 = close / ema120_stock
                        result['a1'] = a1
                        market_midline = self._ema(ema120_stock * a1, 2)
                    else:
                        # 使用真实大盘数据计算
                        # A1 = 大盘收盘价 / EMA(大盘收盘价, 120) （大盘强弱系数）
                        index_close = df['close_index']
                        ema120_index = self._ema(index_close, 120)
                        a1 = index_close / ema120_index
                        result['a1'] = a1
                        # 大盘中线 = EMA(个股EMA120 × 大盘强弱系数, 2)
                        market_midline = self._ema(ema120_stock * a1, 2)
                else:
                    logger.warning("[强势起爆指标] 合并大盘数据失败，使用个股价格作为代理")
                    # 使用个股价格作为代理
                    # 先计算所有指标
                    close = df['Close']
                    open_price = df['Open']
                    high = df['High']
                    low = df['Low']
                    
                    aaa = (3 * close + high + low + open_price) / 6
                    result['aaa'] = aaa
                    var1 = self._ema(aaa, 35)
                    result['var1'] = var1
                    var2 = (self._hhv(var1, 5) + self._hhv(var1, 15) + self._hhv(var1, 30)) / 3
                    result['var2'] = var2
                    var3 = (self._llv(var1, 5) + self._llv(var1, 15) + self._llv(var1, 30)) / 3
                    result['var3'] = var3
                    bull_line = (self._hhv(var2, 5) + self._hhv(var2, 15) + self._hhv(var2, 30)) / 3
                    result['bull_line'] = bull_line
                    bear_line = (self._llv(var3, 5) + self._llv(var3, 15) + self._llv(var3, 30)) / 3
                    result['bear_line'] = bear_line
                    ema120_stock = self._ema(close, 120)
                    
                    a1 = close / ema120_stock
                    result['a1'] = a1
                    market_midline = self._ema(ema120_stock * a1, 2)
                    
            except Exception as e:
                logger.warning(f"[强势起爆指标] 使用大盘数据计算失败: {e}，使用个股价格作为代理")
                # 回退到代理模式
                # 先计算所有指标
                close = df['Close']
                open_price = df['Open']
                high = df['High']
                low = df['Low']
                
                aaa = (3 * close + high + low + open_price) / 6
                result['aaa'] = aaa
                var1 = self._ema(aaa, 35)
                result['var1'] = var1
                var2 = (self._hhv(var1, 5) + self._hhv(var1, 15) + self._hhv(var1, 30)) / 3
                result['var2'] = var2
                var3 = (self._llv(var1, 5) + self._llv(var1, 15) + self._llv(var1, 30)) / 3
                result['var3'] = var3
                bull_line = (self._hhv(var2, 5) + self._hhv(var2, 15) + self._hhv(var2, 30)) / 3
                result['bull_line'] = bull_line
                bear_line = (self._llv(var3, 5) + self._llv(var3, 15) + self._llv(var3, 30)) / 3
                result['bear_line'] = bear_line
                ema120_stock = self._ema(close, 120)
                
                a1 = close / ema120_stock
                result['a1'] = a1
                market_midline = self._ema(ema120_stock * a1, 2)
            
        else:
            logger.warning("[强势起爆指标] 未获取到真实大盘指数数据，使用个股价格作为代理")
            
            # 直接计算所有指标
            close = df['Close']
            open_price = df['Open']
            high = df['High']
            low = df['Low']
            
            aaa = (3 * close + high + low + open_price) / 6
            result['aaa'] = aaa
            var1 = self._ema(aaa, 35)
            result['var1'] = var1
            var2 = (self._hhv(var1, 5) + self._hhv(var1, 15) + self._hhv(var1, 30)) / 3
            result['var2'] = var2
            var3 = (self._llv(var1, 5) + self._llv(var1, 15) + self._llv(var1, 30)) / 3
            result['var3'] = var3
            bull_line = (self._hhv(var2, 5) + self._hhv(var2, 15) + self._hhv(var2, 30)) / 3
            result['bull_line'] = bull_line
            bear_line = (self._llv(var3, 5) + self._llv(var3, 15) + self._llv(var3, 30)) / 3
            result['bear_line'] = bear_line
            ema120_stock = self._ema(close, 120)
            
            # A1:= 个股收盘价 / EMA(个股收盘价, 120) （模拟大盘强弱系数）
            a1 = close / ema120_stock
            result['a1'] = a1

            # 大盘中线: EMA(个股EMA120 × A1, 2)
            market_midline = self._ema(ema120_stock * a1, 2)
        
        result['market_midline'] = market_midline

        # 强势: AAA>=牛线 AND AAA>=大盘中线
        strong = (aaa >= bull_line) & (aaa >= market_midline)
        result['strong'] = strong

        # 次强: AAA<牛线 AND AAA>=大盘中线 AND 牛线>大盘中线
        sub_strong = (aaa < bull_line) & (aaa >= market_midline) & (bull_line > market_midline)
        result['sub_strong'] = sub_strong

        # 弱势: (AAA<=大盘中线 AND AAA<=牛线) OR (AAA>=牛线 AND AAA<大盘中线 AND 牛线<大盘中线)
        weak = ((aaa <= market_midline) & (aaa <= bull_line)) | (
            (aaa >= bull_line) & (aaa < market_midline) & (bull_line < market_midline)
        )
        result['weak'] = weak
        
        # 红色箱体：个股高于大盘中线低于牛线
        # 箱体：底部为大盘中线，顶部为aaa
        red_box_open = np.where((aaa > market_midline) & (aaa < bull_line), market_midline, np.nan)
        red_box_high = np.where((aaa > market_midline) & (aaa < bull_line), aaa, np.nan)
        red_box_low = np.where((aaa > market_midline) & (aaa < bull_line), market_midline, np.nan)
        red_box_close = np.where((aaa > market_midline) & (aaa < bull_line), aaa, np.nan)
        
        result['red_box_open'] = red_box_open
        result['red_box_high'] = red_box_high
        result['red_box_low'] = red_box_low
        result['red_box_close'] = red_box_close
        
        # 紫色箱体：个股同时高于牛线和大盘中线
        # 箱体：底部为max(牛线, 大盘中线)，顶部为aaa
        max_line = np.maximum(bull_line, market_midline)
        purple_box_open = np.where((aaa > bull_line) & (aaa > market_midline), max_line, np.nan)
        purple_box_high = np.where((aaa > bull_line) & (aaa > market_midline), aaa, np.nan)
        purple_box_low = np.where((aaa > bull_line) & (aaa > market_midline), max_line, np.nan)
        purple_box_close = np.where((aaa > bull_line) & (aaa > market_midline), aaa, np.nan)
        
        result['purple_box_open'] = purple_box_open
        result['purple_box_high'] = purple_box_high
        result['purple_box_low'] = purple_box_low
        result['purple_box_close'] = purple_box_close

        return result
