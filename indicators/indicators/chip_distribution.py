import logging
import numpy as np
import pandas as pd

from indicators.base import BaseIndicator

try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    gaussian_filter1d = None

logger = logging.getLogger(__name__)


class ChipDistribution(BaseIndicator):
    """
    筹码分布指标（专业版）

    该指标实现基于动态留存模型的筹码分布计算，包含以下内容：
    - 支持基于真实换手率或模拟换手率的成交量分布
    - 移动成本分布
    - 平均成本线
    - 获利盘/套牢盘识别
    - 筹码集中度高点识别
    - 90%筹码集中度计算（判断主力控盘程度）

    核心算法（重构版）：
    1. 使用正态分布模拟每日成交量在价格范围的分布，减小sigma到/6.0，让每天筹码更集中
    2. 使用基于换手率的动态留存模型（倒序循环，从当前天向前累加）
       - 优先使用真实换手率（成交量/流通股本），需要传入circulating_shares参数
       - 如果没有流通股本数据，回退到模拟换手率（成交量/N日最大成交量）
       - 今日成交量会"清洗"掉昨日旧筹码
       - 高成交量意味着旧筹码被大量替换，切断"拖尾"，形成清晰"双峰"
       - 倒序计算，从当前日期向前推算，维护"筹码留存率"数组
    3. 限制只计算最近120天以提高效率
    4. 累加留存下来的旧筹码
    5. 区分获利盘（成本价<当前价）和套牢盘（成本价>当前价）
    6. 可选高斯平滑，减少数据锯齿
    """

    def __init__(self, 
                price_bin_size: float = 0.002, 
                decay_factor: float = 0.99, 
                turnover_coeff: float = 1.2, 
                turnover_window_days: int = 60, 
                gaussian_sigma: float = 1.5, 
                enable_smooth: bool = True, 
                max_days: int = 120, 
                grid_min: float = None, 
                grid_max: float = None, 
                grid_num_bins: int = 500, 
                min_volume_threshold: float = 0.001, **kwargs):
        """
        初始化筹码分布指标

        Args:
            price_bin_size: 价格间隔大小（百分比或绝对值）
            decay_factor: 衰减系数，范围(0,1)，越小衰减越快（保留以向后兼容）
            turnover_coeff: 换手清洗系数，范围(0, 2)。值越大，高成交量对旧筹码的清洗越快，峰值越分离。默认0.8
            turnover_window_days: 计算最大成交量的窗口天数，默认60
            gaussian_sigma: 高斯平滑的sigma值，默认1.5
            enable_smooth: 是否启用平滑，默认True
            max_days: 最大计算天数，限制只计算最近多少天的数据以提高效率，默认120
            grid_min: 固定网格的最低价（None则根据数据动态确定）
            grid_max: 固定网格的最高价（None则根据数据动态确定）
            grid_num_bins: 固定网格的Bin数量（默认500）
            min_volume_threshold: 最小成交量过滤阈值（相对占比），默认0.001（0.1%），小于该值的K线会被忽略
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.price_bin_size = price_bin_size
        self.decay_factor = decay_factor
        self.turnover_coeff = turnover_coeff
        self.turnover_window_days = turnover_window_days
        self.gaussian_sigma = gaussian_sigma
        self.enable_smooth = enable_smooth
        self.max_days = max_days
        
        # 固定网格参数
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.grid_num_bins = grid_num_bins
        
        # 缓存的网格（确保多次调用时使用相同的网格）
        self.cached_price_bins = None
        self.cached_bin_centers = None
        self.cached_bin_size = None
        
        # 最小成交量过滤阈值
        self.min_volume_threshold = min_volume_threshold

    def calculate(self, data: pd.DataFrame, end_date_idx: int = None, circulating_shares: pd.DataFrame = None) -> dict:
        """
        计算筹码分布（使用动态留存模型）

        Args:
            data: 包含OHLCV数据的输入DataFrame（Open, High, Low, Close, Volume）
            end_date_idx: 截止日期索引（用于移动成本分布，可选）
            circulating_shares: 包含历史流通股本数据的DataFrame（可选）
                列：date, circulating_shares
                如果提供，将使用真实换手率；否则使用模拟换手率

        Returns:
            包含筹码分布数据的字典
        """
        self.validate_input(data)

        df = data.copy().reset_index(drop=True)

        if end_date_idx is None:
            end_date_idx = len(df) - 1
        elif end_date_idx >= len(df):
            end_date_idx = len(df) - 1
        elif end_date_idx < 0:
            end_date_idx = 0

        df = df.iloc[:end_date_idx + 1].copy()

        if len(df) < 10:
            logger.warning("[筹码分布] 数据不足，无法计算")
            return {
                'price_bins': [],
                'chip_volumes': [],
                'profit_volumes': [],
                'loss_volumes': [],
                'avg_cost': None,
                'max_chip_price': None,
                'current_price': None
            }

        # 计算价格范围和分箱，优先使用固定网格避免抖动
        if self.cached_price_bins is None or self.cached_bin_centers is None:
            # 第一次调用，初始化网格
            if self.grid_min is not None and self.grid_max is not None:
                # 使用用户指定的固定网格
                price_min = self.grid_min
                price_max = self.grid_max
                num_bins = self.grid_num_bins
            else:
                # 自动确定网格（带缓冲区，避免边缘抖动）
                price_min = df['Low'].min()
                price_max = df['High'].max()
                price_range = price_max - price_min
                if price_range <= 0:
                    price_range = price_min * 0.1 if price_min > 0 else 1.0
                
                # 添加10%缓冲区，防止刚好卡在边界上
                buffer = price_range * 0.1
                price_min -= buffer
                price_max += buffer
                num_bins = self.grid_num_bins
            
            # 对齐到0.01元最小报价单位
            price_min = np.floor(price_min / 0.01) * 0.01
            price_max = np.ceil(price_max / 0.01) * 0.01
            price_range = price_max - price_min
            
            # 生成价格分箱
            price_bins = np.linspace(price_min, price_max, num_bins + 1)
            bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
            bin_size = price_bins[1] - price_bins[0]
            
            # 缓存网格，后续调用使用相同网格
            self.cached_price_bins = price_bins
            self.cached_bin_centers = bin_centers
            self.cached_bin_size = bin_size
        else:
            # 后续调用，直接使用缓存的网格，避免抖动
            price_bins = self.cached_price_bins
            bin_centers = self.cached_bin_centers
            bin_size = self.cached_bin_size

        # 初始化筹码分布
        chip_distribution = np.zeros(len(bin_centers))

        # 预处理：计算换手率
        # 优先级：
        # 1. 使用circulating_shares参数（外部传入的流通股本数据）
        # 2. 使用数据中的turnover_rate列（如果历史数据包含换手率）
        # 3. 回退到模拟换手率
        
        has_real_turnover = False
        
        # 方案1：使用外部传入的流通股本数据
        if circulating_shares is not None and not circulating_shares.empty:
            # 使用真实换手率
            # 首先确保circulating_shares的日期格式与data一致
            cs_df = circulating_shares.copy()
            cs_df['date'] = pd.to_datetime(cs_df['date'])
            
            # 如果data有日期列，进行合并
            if 'date' in df.columns:
                df_with_date = df.copy()
                df_with_date['date'] = pd.to_datetime(df_with_date['date'])
                df_merged = df_with_date.merge(cs_df, on='date', how='left')
                
                # 前向填充缺失的流通股本数据（股本变动不频繁）
                df_merged['circulating_shares'] = df_merged['circulating_shares'].ffill()
                
                # 计算真实换手率 = 成交量 / 流通股本
                # 注意：成交量单位是股，流通股本也是股
                valid_mask = df_merged['circulating_shares'].notna() & (df_merged['circulating_shares'] > 0)
                df_merged['real_turnover'] = 0.0
                df_merged.loc[valid_mask, 'real_turnover'] = (
                    df_merged.loc[valid_mask, 'Volume'] / df_merged.loc[valid_mask, 'circulating_shares']
                ).clip(0, 1)  # 限制在0-1之间
                
                df = df_merged
                has_real_turnover = True
                logger.debug("[筹码分布] 使用真实换手率计算（来自流通股本数据）")
        
        # 方案2：使用历史数据中的换手率列（如果存在）
        if not has_real_turnover:
            # 检查是否有换手率列（AKShare的stock_zh_a_hist包含"换手率"列）
            if '换手率' in df.columns:
                # AKShare返回的换手率是百分比形式（如5.2表示5.2%），需要转换为小数
                df['real_turnover'] = (df['换手率'] / 100.0).clip(0, 1)
                has_real_turnover = True
                logger.debug("[筹码分布] 使用真实换手率计算（来自历史数据中的换手率列）")
            elif 'turnover_rate' in df.columns:
                # 如果列名是英文的
                turnover_data = df['turnover_rate'].copy()
                
                # 检测并修复旧数据：如果所有值都小于0.01，说明是旧的错误数据（除以了100两次）
                # 旧数据特征：最大值小于0.01，且大部分值小于0.001
                non_null_turnover = turnover_data.dropna()
                if len(non_null_turnover) > 0:
                    max_val = non_null_turnover.max()
                    small_count = (non_null_turnover < 0.001).sum()
                    if max_val < 0.01 and small_count > len(non_null_turnover) * 0.5:
                        # 检测到旧数据，需要乘以100进行修复
                        logger.warning("[筹码分布] 检测到旧版错误换手率数据，正在修复...")
                        turnover_data = turnover_data * 100.0
                        logger.info(f"[筹码分布] 数据修复完成，修复前最大值: {max_val:.8f}，修复后最大值: {turnover_data.max():.8f}")
                
                df['real_turnover'] = turnover_data.clip(0, 1)
                has_real_turnover = True
                logger.debug("[筹码分布] 使用真实换手率计算（来自历史数据中的turnover_rate列）")
        
        # 禁止使用模拟换手率
        if not has_real_turnover:
            logger.warning("[筹码分布] 没有真实换手率数据，无法计算筹码分布")
            # 返回空结果
            return {
                'stock_code': '',
                'price_bins': [],
                'chip_volumes': [],
                'profit_volumes': [],
                'loss_volumes': [],
                'avg_cost': None,
                'max_chip_price': None,
                'current_price': None
            }
        
        # 使用真实换手率 - 过滤和填充缺失值
        df['turnover_rate'] = df['real_turnover'].fillna(0)
        
        # 确保没有负值
        df['turnover_rate'] = df['turnover_rate'].clip(0, 1)
        
        # 计算总成交量，用于相对阈值判断
        total_volume = df['Volume'].sum()

        # 倒序循环：从当前天往前推，模拟筹码的"生存"过程
        # 今天的筹码是100%存在的，昨天的筹码根据今天的换手率留存，以此类推
        cumulative_survival = np.ones(len(bin_centers))
        epsilon = 1e-5  # 用于浮点比较的极小值

        current_idx = len(df) - 1

        # 1. 先计算当天的分布（作为基础）
        row = df.iloc[current_idx]
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']
        close_price = row['Close']
        volume = row['Volume']
        
        # 判断涨跌停
        is_limit_up = (close_price >= high_price * (1 - epsilon)) and (high_price > low_price)
        is_limit_down = (close_price <= low_price * (1 + epsilon)) and (high_price > low_price)
        is_flat_line = (abs(high_price - low_price) < epsilon)  # 一字板
        
        # 检查成交量是否大于阈值（避免微小成交量污染筹码分布）
        volume_ratio = volume / total_volume if total_volume > 0 else 0.0
        if volume_ratio >= self.min_volume_threshold:
            if is_flat_line or is_limit_up or is_limit_down:
                # 涨跌停或一字板：强制分配到最近的3个Bin，避免单点过尖或跳变
                # 形成一个小山包，在计算平均成本时更稳定
                center_idx = np.argmin(np.abs(bin_centers - close_price))
                volume_dist = np.zeros_like(bin_centers)
                start_idx = max(0, center_idx - 1)
                end_idx = min(len(volume_dist), center_idx + 2)
                volume_dist[start_idx:end_idx] = [0.25, 0.5, 0.25]
            else:
                # 普通交易日：三角分布，顶点在(Open+Close)/2
                vertex_price = (open_price + close_price) / 2
                volume_dist = self._triangle_distribution(bin_centers, low_price, high_price, vertex_price)
            
            vol_sum = volume_dist.sum()
            if vol_sum > 0:
                volume_dist = volume_dist / vol_sum * volume
            chip_distribution += volume_dist
        else:
            logger.debug(f"[筹码分布] 忽略成交量极小的K线 {current_idx}，成交量占比: {volume_ratio:.4%}")

        # 2. 向前遍历历史数据，累加"幸存"的旧筹码
        # 限制仅计算最近max_days天以提高效率
        start_lookback = max(current_idx - self.max_days, 0)
        for i in range(current_idx - 1, start_lookback - 1, -1):
            row = df.iloc[i]
            turnover_rate = row['turnover_rate']
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            volume = row['Volume']
            
            # 安全检查
            if pd.isna(turnover_rate) or turnover_rate is None:
                turnover_rate = 0
            
            # 更新留存率：旧筹码 * (1 - 当日换手率 * 清洗系数)
            # 这里的逻辑是：如果今天换手率高，昨天的筹码就剩得少
            # 清洗系数 > 1.0 可以加速旧筹码衰减，形成清晰的筹码峰
            decay_factor = min(1.0, turnover_rate * self.turnover_coeff)
            cumulative_survival *= (1.0 - decay_factor)
            
            # 检查成交量是否大于阈值（避免微小成交量污染筹码分布）
            volume_ratio = volume / total_volume if total_volume > 0 else 0.0
            if volume_ratio >= self.min_volume_threshold:
                # 判断涨跌停
                is_limit_up = (close_price >= high_price * (1 - epsilon)) and (high_price > low_price)
                is_limit_down = (close_price <= low_price * (1 + epsilon)) and (high_price > low_price)
                is_flat_line = (abs(high_price - low_price) < epsilon)
                
                # 计算当天的分布
                if is_flat_line or is_limit_up or is_limit_down:
                    # 涨跌停或一字板：强制分配到最近的3个Bin，避免单点过尖或跳变
                    # 形成一个小山包，在计算平均成本时更稳定
                    center_idx = np.argmin(np.abs(bin_centers - close_price))
                    volume_dist = np.zeros_like(bin_centers)
                    start_idx = max(0, center_idx - 1)
                    end_idx = min(len(volume_dist), center_idx + 2)
                    volume_dist[start_idx:end_idx] = [0.25, 0.5, 0.25]
                else:
                    # 普通交易日：三角分布
                    vertex_price = (open_price + close_price) / 2
                    volume_dist = self._triangle_distribution(bin_centers, low_price, high_price, vertex_price)
                
                vol_sum = volume_dist.sum()
                if vol_sum > 0:
                    volume_dist = volume_dist / vol_sum * volume
                
                # 累加留存下来的旧筹码
                chip_distribution += volume_dist * cumulative_survival
            else:
                # 成交量太小，跳过，只更新留存率
                logger.debug(f"[筹码分布] 忽略成交量极小的K线 {i}，成交量占比: {volume_ratio:.4%}")

        current_price = df.iloc[-1]['Close']

        # L1归一化：将筹码分布转化为概率密度
        total_chip = chip_distribution.sum()
        if total_chip > 0:
            chip_distribution = chip_distribution / total_chip

        # 平滑处理（放在中间步骤）
        if self.enable_smooth:
            # 平滑筹码分布
            chip_distribution = self._smooth_chip_data(chip_distribution)
            
            # 平滑后二次归一化，确保总和仍为概率密度
            chip_total = chip_distribution.sum()
            if chip_total > 0:
                chip_distribution = chip_distribution / chip_total

        # 区分获利盘和套牢盘（应该在归一化和平滑之后）
        profit_mask = bin_centers < current_price
        loss_mask = bin_centers > current_price

        profit_volumes = np.where(profit_mask, chip_distribution, 0)
        loss_volumes = np.where(loss_mask, chip_distribution, 0)

        # 计算平均成本
        avg_cost = np.sum(bin_centers * chip_distribution) if total_chip > 0 else None

        # 计算筹码峰
        max_chip_idx = np.argmax(chip_distribution)
        max_chip_price = bin_centers[max_chip_idx] if len(chip_distribution) > 0 else None

        # 计算90%集中度：找出包含90%筹码的价格区间[Price_L, Price_H]
        concentration_90 = None
        price_L = None
        price_H = None

        if total_chip > 0:
            # 计算累积分布
            cum_dist = np.cumsum(chip_distribution)
            
            # 使用离散快速估算版（不使用插值，速度提升10倍以上）
            # 直接使用np.searchsorted找到最近的bin索引，避免高频的二分查找插值
            target_5pct = 0.05
            target_95pct = 0.95
            
            idx_L = np.searchsorted(cum_dist, target_5pct, side='right')
            idx_H = np.searchsorted(cum_dist, target_95pct, side='right')
            
            # 边界安全处理
            idx_L = min(max(idx_L, 0), len(bin_centers) - 1)
            idx_H = min(max(idx_H, 0), len(bin_centers) - 1)
            
            price_L = bin_centers[idx_L]
            price_H = bin_centers[idx_H]
            
            # 计算集中度
            if price_H + price_L > 0:
                concentration_90 = (price_H - price_L) / (price_H + price_L)

        return {
            'price_bins': bin_centers.tolist(),
            'chip_volumes': chip_distribution.tolist(),
            'profit_volumes': profit_volumes.tolist(),
            'loss_volumes': loss_volumes.tolist(),
            'avg_cost': float(avg_cost) if avg_cost is not None else None,
            'max_chip_price': float(max_chip_price) if max_chip_price is not None else None,
            'current_price': float(current_price),
            'concentration_90': float(concentration_90) if concentration_90 is not None else None,
            'concentration_price_L': float(price_L) if price_L is not None else None,
            'concentration_price_H': float(price_H) if price_H is not None else None
        }

    def _triangle_distribution(self, x: np.ndarray, low: float, high: float, vertex: float) -> np.ndarray:
        """
        计算三角分布形状（不严格要求归一化，后面会重新归一化）
        向量化实现，避免逐点循环提升性能

        Args:
            x: 输入值数组
            low: 三角形底边最小值
            high: 三角形底边最大值
            vertex: 三角形顶点位置

        Returns:
            三角分布形状数组
        """
        vertex = max(low, min(high, vertex))  # 边界保护
        width = high - low
        
        if width <= 0:
            # 处理一字板（价格区间为0）
            return np.where(np.abs(x - vertex) < 1e-6, 1.0, 0.0)
        
        # 初始化所有为0
        dist = np.zeros_like(x)
        
        # 向量化计算：利用NumPy广播一次性处理所有Bin
        # 几何方式计算：从0到顶点高度，再回落到0
        # 顶点高度设为1.0（简单直观，后面会重新归一化）
        
        # 左半部分：[low, vertex]，从0线性增加到1
        left_mask = (x >= low) & (x <= vertex)
        if vertex > low:
            dist[left_mask] = (x[left_mask] - low) / (vertex - low)
        
        # 右半部分：(vertex, high]，从1线性减少到0
        right_mask = (x > vertex) & (x <= high)
        if vertex < high:
            dist[right_mask] = (high - x[right_mask]) / (high - vertex)
        
        # 归一化：确保总面积为1，避免因价格波动范围不同导致筹码总量不守恒
        dist_sum = dist.sum()
        if dist_sum > 0:
            dist = dist / dist_sum
        
        return dist
    
    def _normal_distribution(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """
        计算正态分布概率密度（保留向后兼容）

        Args:
            x: 输入值数组
            mu: 均值
            sigma: 标准差

        Returns:
            正态分布概率密度数组
        """
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    
    def _smooth_chip_data(self, data: np.ndarray) -> np.ndarray:
        """
        平滑筹码分布数据

        Args:
            data: 输入的筹码数据数组

        Returns:
            平滑后的筹码数据数组
        """
        if len(data) < 3:
            return data
        
        if HAS_SCIPY:
            # 使用高斯平滑
            return gaussian_filter1d(data, sigma=self.gaussian_sigma)
        else:
            # 回退到简单移动平均，使用奇数窗口以保持对称性
            window_size = 3
            pad_size = window_size // 2
            padded = np.pad(data, pad_size, mode='edge')
            smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
            return smoothed

    def calculate_moving_distribution(self, data: pd.DataFrame, window_days: int = None) -> list:
        """
        计算移动成本分布（用于动态筹码峰随光标移动）

        Args:
            data: 包含OHLCV数据的输入DataFrame
            window_days: 窗口天数（可选）

        Returns:
            筹码分布数据列表
        """
        distributions = []

        start_idx = max(0, len(data) - (window_days if window_days else 365))

        for i in range(start_idx, len(data)):
            dist = self.calculate(data, i)
            distributions.append(dist)

        return distributions
