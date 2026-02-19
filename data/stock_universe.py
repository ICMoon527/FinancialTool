import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from logger import log
from data.data_fetcher import data_fetcher, HAS_AKSHARE, HAS_TUSHARE
from utils.progress_bar import create_progress_bar

try:
    from data.database import db_manager, HAS_SQLALCHEMY, StockBasic
except ImportError:
    HAS_SQLALCHEMY = False
    db_manager = None
    StockBasic = None


class StockUniverse:
    
    def __init__(self):
        self.all_stocks = None
        self.stock_data = {}
        
        self.core_stock_pool = self._create_core_pool()
        
        self.risk_level_configs = {
            'low': {
                'name': '保守型',
                'description': '适合初入股市的投资者，仅包含上证和深证主板',
                'boards': ['sh_main', 'sz_main'],
                'default_count': 100
            },
            'medium': {
                'name': '稳健型',
                'description': '适合有一定经验的投资者，包含主板和创业板',
                'boards': ['sh_main', 'sz_main', 'gem'],
                'default_count': 200
            },
            'high': {
                'name': '进取型',
                'description': '适合经验丰富的投资者，包含主板、创业板、科创板',
                'boards': ['sh_main', 'sz_main', 'gem', 'star'],
                'default_count': 300
            },
            'all': {
                'name': '激进型',
                'description': '适合风险承受能力极强的投资者，包含所有板块',
                'boards': ['sh_main', 'sz_main', 'gem', 'star', 'bse'],
                'default_count': 500
            }
        }
    
    def _create_core_pool(self) -> List[Tuple[str, str]]:
        pool = [
            ('000001.SZ', '平安银行'),
            ('000002.SZ', '万科A'),
            ('600000.SH', '浦发银行'),
            ('600036.SH', '招商银行'),
            ('600519.SH', '贵州茅台'),
            ('000858.SZ', '五粮液'),
            ('002594.SZ', '比亚迪'),
            ('300750.SZ', '宁德时代'),
            ('601318.SH', '中国平安'),
            ('601899.SH', '紫金矿业'),
            ('000651.SZ', '格力电器'),
            ('000333.SZ', '美的集团'),
            ('600900.SH', '长江电力'),
            ('601888.SH', '中国中免'),
            ('600276.SH', '恒瑞医药'),
            ('300015.SZ', '爱尔眼科'),
            ('601012.SH', '隆基绿能'),
            ('600089.SH', '特变电工'),
            ('002475.SZ', '立讯精密'),
            ('002415.SZ', '海康威视'),
            ('601398.SH', '工商银行'),
            ('601857.SH', '中国石油'),
            ('600028.SH', '中国石化'),
            ('600030.SH', '中信证券'),
            ('000725.SZ', '京东方A'),
            ('600585.SH', '海螺水泥'),
            ('600887.SH', '伊利股份'),
            ('300124.SZ', '汇川技术'),
            ('300760.SZ', '迈瑞医疗'),
            ('600309.SH', '万华化学'),
            ('600690.SH', '海尔智家'),
            ('300274.SZ', '阳光电源'),
            ('601328.SH', '交通银行'),
            ('601166.SH', '兴业银行'),
            ('600016.SH', '民生银行'),
            ('601939.SH', '建设银行'),
            ('601288.SH', '农业银行'),
            ('601988.SH', '中国银行'),
            ('000568.SZ', '泸州老窖'),
            ('600809.SH', '山西汾酒'),
            ('000596.SZ', '古井贡酒'),
            ('600779.SH', '水井坊'),
            ('603369.SH', '今世缘'),
            ('002304.SZ', '洋河股份'),
            ('600600.SH', '青岛啤酒'),
            ('000895.SZ', '双汇发展'),
            ('600059.SH', '古越龙山'),
            ('000921.SZ', '海信家电'),
            ('002508.SZ', '老板电器'),
            ('002242.SZ', '九阳股份'),
            ('300142.SZ', '沃森生物'),
            ('300003.SZ', '乐普医疗'),
            ('300244.SZ', '迪安诊断'),
            ('300122.SZ', '智飞生物'),
            ('300347.SZ', '泰格医药'),
            ('600196.SH', '复星医药'),
            ('600518.SH', '康美药业'),
            ('600521.SH', '华海药业'),
            ('600812.SH', '华北制药'),
            ('601607.SH', '上海医药'),
            ('000423.SZ', '东阿阿胶'),
            ('000538.SZ', '云南白药'),
            ('000999.SZ', '华润三九'),
            ('002007.SZ', '华兰生物'),
            ('002422.SZ', '科伦药业'),
        ]
        return pool
    
    def _is_sh_main(self, ts_code: str) -> bool:
        symbol = ts_code.split('.')[0]
        return symbol.startswith('600') or symbol.startswith('601') or symbol.startswith('603') or symbol.startswith('605')
    
    def _is_sz_main(self, ts_code: str) -> bool:
        symbol = ts_code.split('.')[0]
        return symbol.startswith('000') or symbol.startswith('001') or symbol.startswith('002') or symbol.startswith('003')
    
    def _is_gem(self, ts_code: str) -> bool:
        symbol = ts_code.split('.')[0]
        return symbol.startswith('300') or symbol.startswith('301')
    
    def _is_star(self, ts_code: str) -> bool:
        symbol = ts_code.split('.')[0]
        return symbol.startswith('688')
    
    def _is_bse(self, ts_code: str) -> bool:
        symbol = ts_code.split('.')[0]
        return symbol.startswith('8') or symbol.startswith('9')
    
    def _filter_by_boards(self, df: pd.DataFrame, boards: List[str]) -> pd.DataFrame:
        mask = pd.Series([False] * len(df), index=df.index)
        
        for board in boards:
            if board == 'sh_main':
                mask = mask | df['ts_code'].apply(self._is_sh_main)
            elif board == 'sz_main':
                mask = mask | df['ts_code'].apply(self._is_sz_main)
            elif board == 'gem':
                mask = mask | df['ts_code'].apply(self._is_gem)
            elif board == 'star':
                mask = mask | df['ts_code'].apply(self._is_star)
            elif board == 'bse':
                mask = mask | df['ts_code'].apply(self._is_bse)
        
        return df[mask].copy()
    
    def load_stock_list_from_db(self) -> Optional[pd.DataFrame]:
        """从数据库加载股票基本信息"""
        try:
            if not HAS_SQLALCHEMY or not db_manager or not StockBasic:
                return None
            
            session = db_manager.get_session()
            try:
                stocks = session.query(StockBasic).all()
                if not stocks:
                    return None
                
                data = []
                for stock in stocks:
                    data.append({
                        'ts_code': stock.ts_code,
                        'symbol': stock.ts_code.split('.')[0],
                        'name': stock.name,
                        'area': stock.area,
                        'industry': stock.industry,
                        'market': stock.market,
                        'list_date': stock.list_date
                    })
                
                df = pd.DataFrame(data)
                log.info(f"从数据库加载 {len(df)} 支股票基本信息")
                return df
            finally:
                session.close()
        except Exception as e:
            log.warning(f"从数据库加载股票列表失败: {e}")
            return None
    
    def fetch_all_stock_list(self, force_refresh: bool = False, basic_only: bool = False, use_cache: bool = True) -> pd.DataFrame:
        if self.all_stocks is not None and not force_refresh:
            return self.all_stocks
        
        try:
            if use_cache and not force_refresh:
                log.info("尝试从数据库加载股票列表...")
                cached_df = self.load_stock_list_from_db()
                if cached_df is not None and not cached_df.empty:
                    self.all_stocks = cached_df
                    log.info(f"使用缓存的股票列表，共 {len(cached_df)} 支")
                    return cached_df
            
            log.info("正在获取所有A股股票列表...")
            if basic_only:
                df = data_fetcher.fetch_stock_list_basic()
            else:
                df = data_fetcher.fetch_stock_list()
            
            self.all_stocks = df
            
            try:
                data_fetcher.save_stock_list(df)
            except Exception as save_error:
                log.warning(f"保存股票列表到数据库失败: {save_error}")
            
            log.info(f"成功获取 {len(df)} 支A股股票")
            return df
            
        except Exception as e:
            log.error(f"获取所有A股股票列表失败: {e}")
            raise
    
    def get_stock_pool(self, pool_type: str = 'core', size: Optional[int] = None, 
                       boards: Optional[List[str]] = None, basic_only: bool = False, use_cache: bool = True) -> List[Tuple[str, str]]:
        if pool_type == 'core':
            if size and size > 0:
                return self.core_stock_pool[:size]
            return self.core_stock_pool
        
        elif pool_type in self.risk_level_configs:
            try:
                df = self.fetch_all_stock_list(basic_only=basic_only, use_cache=use_cache)
                config = self.risk_level_configs[pool_type]
                filtered_df = self._filter_by_boards(df, config['boards'])
                
                filtered_df = filtered_df[~filtered_df['name'].str.contains('ST', na=False)]
                filtered_df = filtered_df[~filtered_df['name'].str.contains(r'\*ST', na=False)]
                
                stocks = list(zip(filtered_df['ts_code'], filtered_df['name']))
                
                sh_count = sum(1 for ts_code, _ in stocks if self._is_sh_main(ts_code))
                sz_count = sum(1 for ts_code, _ in stocks if self._is_sz_main(ts_code))
                gem_count = sum(1 for ts_code, _ in stocks if self._is_gem(ts_code))
                star_count = sum(1 for ts_code, _ in stocks if self._is_star(ts_code))
                bse_count = sum(1 for ts_code, _ in stocks if self._is_bse(ts_code))
                
                log.info(f"股票池筛选完成: 上证{sh_count}支, 深证{sz_count}支, 创业板{gem_count}支, 科创板{star_count}支, 北交所{bse_count}支, 总计{len(stocks)}支")
                
                if size and size > 0:
                    return stocks[:size]
                return stocks
            except Exception as e:
                log.warning(f"获取风险等级股票池失败，使用核心池: {e}")
                return self.get_stock_pool('core', size, basic_only=basic_only, use_cache=use_cache)
        
        elif pool_type == 'all' or (pool_type == 'custom' and boards):
            try:
                df = self.fetch_all_stock_list(basic_only=basic_only, use_cache=use_cache)
                if boards:
                    filtered_df = self._filter_by_boards(df, boards)
                else:
                    filtered_df = df
                
                filtered_df = filtered_df[~filtered_df['name'].str.contains('ST', na=False)]
                filtered_df = filtered_df[~filtered_df['name'].str.contains(r'\*ST', na=False)]
                
                stocks = list(zip(filtered_df['ts_code'], filtered_df['name']))
                
                if size and size > 0:
                    return stocks[:size]
                return stocks
            except Exception as e:
                log.warning(f"获取全部股票失败，使用核心池: {e}")
                return self.get_stock_pool('core', size, basic_only=basic_only, use_cache=use_cache)
        
        else:
            return self.core_stock_pool
    
    def get_risk_level_info(self, risk_level: str) -> Dict:
        if risk_level in self.risk_level_configs:
            return self.risk_level_configs[risk_level]
        return None
    
    def fetch_all_stock_data(self, pool_type: str = 'core', stock_count: int = 80, 
                            days: int = 300, show_progress: bool = True,
                            batch_size: int = 50) -> Dict[str, pd.DataFrame]:
        stock_data = {}
        
        target_stocks = self.get_stock_pool(pool_type, stock_count)
        
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=300)).strftime('%Y%m%d')
        
        ts_code_list = [ts_code for ts_code, _ in target_stocks]
        name_map = {ts_code: name for ts_code, name in target_stocks}
        
        if show_progress:
            print(f"\n正在批量获取 {len(ts_code_list)} 支股票数据...")
        
        batch_result = data_fetcher.fetch_stock_daily_batch(
            ts_code_list=ts_code_list,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        
        success_count = 0
        total = len(batch_result)
        pb = create_progress_bar(total, '处理股票数据')
        
        for i, (ts_code, df) in enumerate(batch_result.items()):
            if not df.empty:
                df = df.copy()
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').reset_index(drop=True)
                
                if 'vol' in df.columns and 'volume' not in df.columns:
                    df['volume'] = df['vol']
                
                df['name'] = name_map.get(ts_code, ts_code)
                stock_data[ts_code] = df
                success_count += 1
            pb.update(i + 1)
        
        pb.finish()
        
        if show_progress:
            print(f"成功获取 {success_count}/{len(ts_code_list)} 支股票数据")
        
        log.info(f"完成获取 {success_count}/{len(ts_code_list)} 支股票数据")
        return stock_data


stock_universe = StockUniverse()
