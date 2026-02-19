import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from config import Config, user_config
from logger import log
from utils.progress_bar import create_progress_bar

try:
    from data.database import db_manager, HAS_SQLALCHEMY
except ImportError:
    HAS_SQLALCHEMY = False
    db_manager = None
    log.warning("数据库功能不可用")

# 只要有 SQLAlchemy 就导入数据模型，不管数据源是否可用
if HAS_SQLALCHEMY:
    try:
        from data.database import StockBasic, StockDaily
    except ImportError:
        StockBasic = None
        StockDaily = None
        log.warning("数据模型导入失败")
else:
    StockBasic = None
    StockDaily = None

try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False
    log.warning("AKShare 未安装")

try:
    import tushare as ts
    HAS_TUSHARE = True
except ImportError:
    HAS_TUSHARE = False
    log.warning("Tushare 未安装")

class DataFetcher:
    def __init__(self):
        self.ts_pro = None
        self._latest_trading_day = None
        if HAS_TUSHARE and Config.TUSHARE_TOKEN:
            try:
                ts.set_token(Config.TUSHARE_TOKEN)
                self.ts_pro = ts.pro_api()
                log.info("Tushare API 初始化成功")
            except Exception as e:
                log.warning(f"Tushare 初始化失败，将使用 AKShare: {e}")
                self.ts_pro = None
        elif HAS_AKSHARE:
            log.warning("未配置 TUSHARE_TOKEN，将使用 AKShare")
        else:
            log.warning("未安装 AKShare 或 Tushare")
    
    def get_available_latest_trading_day(self) -> str:
        """从数据源获取最新的可用交易日 - 真的获取一个股票的数据"""
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            
            if self.ts_pro:
                # 用Tushare获取平安银行数据
                df = self.ts_pro.daily(ts_code='000001.SZ', start_date=start_date, end_date=end_date)
                if not df.empty:
                    return df['trade_date'].iloc[0]
            
            if HAS_AKSHARE:
                # 用AKShare获取平安银行数据
                df = ak.stock_zh_a_hist(symbol='000001', period='daily', 
                                         start_date=start_date, end_date=end_date, adjust='qfq')
                if not df.empty:
                    return pd.to_datetime(df['日期'].iloc[-1]).strftime('%Y%m%d')
            
            # 都不行的话用今天
            return datetime.now().strftime('%Y%m%d')
        except Exception as e:
            log.warning(f"从数据源获取最新交易日失败: {e}")
            return datetime.now().strftime('%Y%m%d')
    
    def get_latest_trading_day(self) -> str:
        """获取数据库中最新的交易日"""
        if self._latest_trading_day:
            return self._latest_trading_day
        
        try:
            today = datetime.now().strftime('%Y%m%d')
            
            for offset in range(60):
                check_date = (datetime.strptime(today, '%Y%m%d') - timedelta(days=offset)).strftime('%Y%m%d')
                
                df = self.load_stock_daily_from_db('000001.SZ', check_date, check_date)
                
                if not df.empty:
                    self._latest_trading_day = check_date
                    return check_date
            
            return today
        except Exception as e:
            log.warning(f"获取最新交易日失败: {e}")
            return datetime.now().strftime('%Y%m%d')
    
    def fetch_stock_list_basic(self) -> pd.DataFrame:
        """只获取股票基本信息，不获取实时行情数据"""
        try:
            log.info("正在获取股票基本信息列表...")
            
            if HAS_AKSHARE:
                try:
                    all_stocks = []
                    
                    log.info("正在获取上交所股票列表...")
                    df_sh = ak.stock_info_sh_name_code()
                    df_sh['ts_code'] = df_sh['证券代码'] + '.SH'
                    df_sh['symbol'] = df_sh['证券代码']
                    df_sh['name'] = df_sh['证券简称']
                    all_stocks.append(df_sh[['ts_code', 'symbol', 'name']])
                    log.info(f"获取到 {len(df_sh)} 支上交所股票")
                    
                    log.info("正在尝试获取深交所股票列表...")
                    try:
                        df_sz = ak.stock_info_sz_name_code()
                        df_sz['ts_code'] = df_sz['A股代码'] + '.SZ'
                        df_sz['symbol'] = df_sz['A股代码']
                        df_sz['name'] = df_sz['A股简称']
                        all_stocks.append(df_sz[['ts_code', 'symbol', 'name']])
                        log.info(f"获取到 {len(df_sz)} 支深交所股票")
                    except Exception as sz_error:
                        log.warning(f"获取深交所股票列表失败: {sz_error}")
                        log.info("将仅使用上交所股票进行分析")
                    
                    if all_stocks:
                        df = pd.concat(all_stocks, ignore_index=True)
                        df['area'] = None
                        df['industry'] = None
                        df['market'] = None
                        df['list_date'] = None
                        
                        log.info(f"总计获取到 {len(df)} 只A股股票")
                        return df
                    else:
                        raise Exception("未能获取到任何股票数据")
                except Exception as ak_error:
                    log.warning(f"获取股票基本信息失败，尝试其他方法: {ak_error}")
            
            if self.ts_pro:
                df = self.ts_pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,market,list_date')
                log.info(f"Tushare获取到 {len(df)} 只股票")
                return df
            else:
                raise ImportError("需要安装 AKShare 或 Tushare")
            
        except Exception as e:
            log.error(f"获取股票基本信息列表失败: {e}")
            raise
    
    def fetch_stock_list(self) -> pd.DataFrame:
        try:
            log.info("正在获取股票列表...")
            
            if HAS_AKSHARE:
                try:
                    log.info("正在获取A股实时行情列表...")
                    df_spot = ak.stock_zh_a_spot_em()
                    
                    df_spot['ts_code'] = df_spot['代码'].apply(lambda x: f"{x}.SZ" if x.startswith('0') or x.startswith('3') else f"{x}.SH")
                    df_spot['symbol'] = df_spot['代码']
                    df_spot['name'] = df_spot['名称']
                    df_spot['area'] = None
                    df_spot['industry'] = None
                    df_spot['market'] = None
                    df_spot['list_date'] = None
                    
                    df = df_spot[['ts_code', 'symbol', 'name', 'area', 'industry', 'market', 'list_date']]
                    
                    sh_count = sum(1 for code in df_spot['代码'] if code.startswith('6'))
                    sz_count = len(df_spot) - sh_count
                    log.info(f"成功获取到 {len(df)} 只A股股票 (上交所: {sh_count}支, 深交所: {sz_count}支)")
                    return df
                        
                except Exception as ak_error:
                    log.warning(f"通过实时行情获取股票列表失败，尝试其他方法: {ak_error}")
                    
                    try:
                        all_stocks = []
                        
                        log.info("正在获取上交所股票列表...")
                        df_sh = ak.stock_info_sh_name_code()
                        df_sh['ts_code'] = df_sh['证券代码'] + '.SH'
                        df_sh['symbol'] = df_sh['证券代码']
                        df_sh['name'] = df_sh['证券简称']
                        all_stocks.append(df_sh[['ts_code', 'symbol', 'name']])
                        log.info(f"获取到 {len(df_sh)} 支上交所股票")
                        
                        log.info("正在尝试获取深交所股票列表...")
                        try:
                            df_sz = ak.stock_info_sz_name_code()
                            df_sz['ts_code'] = df_sz['A股代码'] + '.SZ'
                            df_sz['symbol'] = df_sz['A股代码']
                            df_sz['name'] = df_sz['A股简称']
                            all_stocks.append(df_sz[['ts_code', 'symbol', 'name']])
                            log.info(f"获取到 {len(df_sz)} 支深交所股票")
                        except Exception as sz_error:
                            log.warning(f"获取深交所股票列表失败: {sz_error}")
                            log.info("将仅使用上交所股票进行分析")
                        
                        if all_stocks:
                            df = pd.concat(all_stocks, ignore_index=True)
                            df['area'] = None
                            df['industry'] = None
                            df['market'] = None
                            df['list_date'] = None
                            
                            log.info(f"总计获取到 {len(df)} 只A股股票")
                            return df
                        else:
                            raise Exception("未能获取到任何股票数据")
                    except Exception as ak_error2:
                        log.warning(f"其他方法也失败，尝试Tushare: {ak_error2}")
            
            if self.ts_pro:
                df = self.ts_pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,market,list_date')
                log.info(f"Tushare获取到 {len(df)} 只股票")
                return df
            else:
                raise ImportError("需要安装 AKShare 或 Tushare")
            
        except Exception as e:
            log.error(f"获取股票列表失败: {e}")
            raise
    
    def save_stock_list(self, df: pd.DataFrame):
        if not HAS_AKSHARE and not HAS_TUSHARE:
            log.warning("数据存储功能需要 AKShare 或 Tushare")
            return
        
        if StockBasic is None:
            log.warning("StockBasic 模型未加载，无法保存数据到数据库")
            return
        
        records = []
        for _, row in df.iterrows():
            record = {
                'ts_code': row['ts_code'],
                'name': row['name'],
                'area': row.get('area'),
                'industry': row.get('industry'),
                'market': row.get('market'),
                'list_date': self._parse_date(row.get('list_date'))
            }
            records.append(record)
        
        db_manager.upsert_daily_data(StockBasic, records, ['ts_code'])
    
    def load_stock_daily_from_db(self, ts_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        try:
            if not HAS_SQLALCHEMY or not db_manager or StockDaily is None:
                return pd.DataFrame()
            
            session = db_manager.get_session()
            try:
                query = session.query(StockDaily).filter(StockDaily.ts_code == ts_code)
                
                if start_date:
                    start_dt = self._parse_date(start_date)
                    if start_dt:
                        query = query.filter(StockDaily.trade_date >= start_dt)
                
                if end_date:
                    end_dt = self._parse_date(end_date)
                    if end_dt:
                        query = query.filter(StockDaily.trade_date <= end_dt)
                
                results = query.order_by(StockDaily.trade_date).all()
                
                if not results:
                    return pd.DataFrame()
                
                data = []
                for row in results:
                    data.append({
                        'ts_code': row.ts_code,
                        'trade_date': row.trade_date.strftime('%Y%m%d'),
                        'open': row.open,
                        'high': row.high,
                        'low': row.low,
                        'close': row.close,
                        'pre_close': row.pre_close,
                        'change': row.change,
                        'pct_chg': row.pct_chg,
                        'vol': row.vol,
                        'amount': row.amount,
                        'volume': row.vol
                    })
                
                df = pd.DataFrame(data)
                return df
            finally:
                session.close()
        except Exception as e:
            log.warning(f"从数据库加载 {ts_code} 数据失败: {e}")
            return pd.DataFrame()
    
    def load_stock_daily_batch_from_db(self, ts_code_list: List[str], start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        批量从数据库加载多只股票的日线数据
        
        Args:
            ts_code_list: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            Dict[ts_code, DataFrame]
        """
        result = {}
        
        if not HAS_SQLALCHEMY or not db_manager or not ts_code_list or StockDaily is None:
            return result
        
        try:
            session = db_manager.get_session()
            try:
                batch_size = 50
                total_batches = (len(ts_code_list) + batch_size - 1) // batch_size
                
                from utils.progress_bar import create_progress_bar
                pb = create_progress_bar(total_batches, '从数据库加载数据')
                
                temp_data = []
                
                for i in range(0, len(ts_code_list), batch_size):
                    batch_codes = ts_code_list[i:i+batch_size]
                    
                    query = session.query(StockDaily).filter(StockDaily.ts_code.in_(batch_codes))
                    
                    if start_date:
                        start_dt = self._parse_date(start_date)
                        if start_dt:
                            query = query.filter(StockDaily.trade_date >= start_dt)
                    
                    if end_date:
                        end_dt = self._parse_date(end_date)
                        if end_dt:
                            query = query.filter(StockDaily.trade_date <= end_dt)
                    
                    batch_results = query.order_by(StockDaily.ts_code, StockDaily.trade_date).all()
                    
                    for row in batch_results:
                        temp_data.append({
                            'ts_code': row.ts_code,
                            'trade_date': row.trade_date.strftime('%Y%m%d'),
                            'open': row.open,
                            'high': row.high,
                            'low': row.low,
                            'close': row.close,
                            'pre_close': row.pre_close,
                            'change': row.change,
                            'pct_chg': row.pct_chg,
                            'vol': row.vol,
                            'amount': row.amount,
                            'volume': row.vol
                        })
                    
                    if pb:
                        pb.update((i // batch_size) + 1)
                
                if not temp_data:
                    return result
                
                df_all = pd.DataFrame(temp_data)
                
                for ts_code in ts_code_list:
                    df_stock = df_all[df_all['ts_code'] == ts_code].copy()
                    if not df_stock.empty:
                        result[ts_code] = df_stock
                
                return result
            finally:
                session.close()
        except Exception as e:
            log.warning(f"批量从数据库加载数据失败: {e}")
            return result
    
    def _is_trading_day_finished(self) -> bool:
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        if now.weekday() >= 5:
            return False
        
        if hour > 15 or (hour == 15 and minute >= 0):
            return True
        
        return False
    
    def check_cache_needs_update(self, ts_code: str, end_date: str = None) -> bool:
        try:
            if not HAS_SQLALCHEMY or not db_manager or StockDaily is None:
                return True
            
            # 获取数据源能提供的最新交易日
            available_latest = self.get_available_latest_trading_day()
            available_latest_dt = self._parse_date(available_latest)
            
            session = db_manager.get_session()
            try:
                records = session.query(StockDaily).filter(
                    StockDaily.ts_code == ts_code
                ).order_by(StockDaily.trade_date).all()
                
                if not records:
                    log.info(f"{ts_code} 数据库中没有数据，需要获取")
                    return True
                
                latest = records[-1]
                earliest = records[0]
                data_count = len(records)
                
                # 检查数据量是否足够（至少需要180个交易日）
                if data_count < 180:
                    log.info(f"{ts_code} 数据量不足（只有{data_count}天，需要至少180天），需要获取更多历史数据")
                    return True
                
                # 检查数据库最新日期是否等于数据源最新日期
                if latest.trade_date != available_latest_dt:
                    log.info(f"{ts_code} 数据源有更新的数据（数据源最新: {available_latest}，数据库最新: {latest.trade_date.strftime('%Y%m%d')}），需要更新")
                    return True
                
                log.info(f"{ts_code} 缓存有效，最新日期: {latest.trade_date}，数据量: {data_count}天")
                return False
            finally:
                session.close()
        except Exception as e:
            log.warning(f"检查缓存状态失败: {e}")
            return True
    
    def fetch_stock_daily(self, ts_code: str, start_date: str = None, end_date: str = None, use_cache: bool = True) -> pd.DataFrame:
        try:
            if not end_date:
                end_date = datetime.now().strftime('%Y%m%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=300)).strftime('%Y%m%d')
            
            if use_cache:
                needs_update = self.check_cache_needs_update(ts_code, end_date)
                
                if not needs_update:
                    df_cache = self.load_stock_daily_from_db(ts_code, start_date, end_date)
                    if not df_cache.empty:
                        return df_cache
            
            log.info(f"正在获取 {ts_code} 的日线数据 ({start_date} - {end_date})...")
            
            df = pd.DataFrame()
            
            if HAS_AKSHARE:
                try:
                    symbol = ts_code.split('.')[0]
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                             start_date=start_date, end_date=end_date, adjust="qfq")
                    df = df.rename(columns={
                        '日期': 'trade_date',
                        '开盘': 'open',
                        '最高': 'high',
                        '最低': 'low',
                        '收盘': 'close',
                        '成交量': 'vol',
                        '成交额': 'amount',
                        '振幅': 'amplitude',
                        '涨跌幅': 'pct_chg',
                        '涨跌额': 'change',
                        '换手率': 'turnover'
                    })
                    if 'vol' in df.columns and 'volume' not in df.columns:
                        df['volume'] = df['vol']
                    df['ts_code'] = ts_code
                    df['pre_close'] = df['close'].shift(1)
                    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')
                    log.info(f"AKShare获取到 {ts_code} {len(df)} 条日线数据")
                except Exception as ak_error:
                    log.warning(f"AKShare获取失败，尝试Tushare: {ak_error}")
                    if self.ts_pro:
                        try:
                            df = self.ts_pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                            log.info(f"Tushare获取到 {ts_code} {len(df)} 条日线数据")
                        except Exception as ts_error:
                            log.error(f"Tushare也失败: {ts_error}")
            elif self.ts_pro:
                df = self.ts_pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                log.info(f"Tushare获取到 {ts_code} {len(df)} 条日线数据")
            else:
                raise ImportError("需要安装 AKShare 或 Tushare")
            
            if not df.empty and use_cache:
                self.save_stock_daily(ts_code, df)
            
            return df
        except Exception as e:
            log.error(f"获取 {ts_code} 日线数据失败: {e}")
            return pd.DataFrame()
    
    def save_stock_daily(self, ts_code: str, df: pd.DataFrame):
        if not HAS_AKSHARE and not HAS_TUSHARE:
            log.warning("数据存储功能需要 AKShare 或 Tushare")
            return
        
        if StockDaily is None:
            log.warning("StockDaily 模型未加载，无法保存数据到数据库")
            return
        
        if df.empty:
            return
        
        records = []
        for _, row in df.iterrows():
            record = {
                'ts_code': row['ts_code'],
                'trade_date': self._parse_date(row['trade_date']),
                'open': self._safe_float(row.get('open')),
                'high': self._safe_float(row.get('high')),
                'low': self._safe_float(row.get('low')),
                'close': self._safe_float(row.get('close')),
                'pre_close': self._safe_float(row.get('pre_close')),
                'change': self._safe_float(row.get('change')),
                'pct_chg': self._safe_float(row.get('pct_chg')),
                'vol': self._safe_float(row.get('vol')),
                'amount': self._safe_float(row.get('amount'))
            }
            records.append(record)
        
        db_manager.upsert_daily_data(StockDaily, records, ['ts_code', 'trade_date'])
    
    def fetch_daily_basic(self, ts_code: str, trade_date: str = None) -> pd.DataFrame:
        try:
            if not self.ts_pro:
                log.warning("需要 Tushare Token 才能获取每日指标")
                return pd.DataFrame()
            
            if trade_date:
                df = self.ts_pro.daily_basic(ts_code=ts_code, trade_date=trade_date)
            else:
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                df = self.ts_pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            log.info(f"获取到 {ts_code} {len(df)} 条每日指标")
            return df
        except Exception as e:
            log.error(f"获取 {ts_code} 每日指标失败: {e}")
            return pd.DataFrame()
    
    def save_daily_basic(self, df: pd.DataFrame):
        if not HAS_AKSHARE and not HAS_TUSHARE:
            log.warning("数据存储功能需要 AKShare 或 Tushare")
            return
        
        if df.empty:
            return
        
        records = []
        for _, row in df.iterrows():
            record = {
                'ts_code': row['ts_code'],
                'trade_date': self._parse_date(row['trade_date']),
                'close': self._safe_float(row.get('close')),
                'turnover_rate': self._safe_float(row.get('turnover_rate')),
                'turnover_rate_f': self._safe_float(row.get('turnover_rate_f')),
                'volume_ratio': self._safe_float(row.get('volume_ratio')),
                'pe': self._safe_float(row.get('pe')),
                'pe_ttm': self._safe_float(row.get('pe_ttm')),
                'pb': self._safe_float(row.get('pb')),
                'ps': self._safe_float(row.get('ps')),
                'ps_ttm': self._safe_float(row.get('ps_ttm')),
                'dv_ratio': self._safe_float(row.get('dv_ratio')),
                'dv_ttm': self._safe_float(row.get('dv_ttm')),
                'total_share': self._safe_float(row.get('total_share')),
                'float_share': self._safe_float(row.get('float_share')),
                'free_share': self._safe_float(row.get('free_share')),
                'total_mv': self._safe_float(row.get('total_mv')),
                'circ_mv': self._safe_float(row.get('circ_mv'))
            }
            records.append(record)
        
        db_manager.upsert_daily_data(StockDailyBasic, records, ['ts_code', 'trade_date'])
    
    def fetch_index_daily(self, ts_code: str = '000001.SH', start_date: str = None, end_date: str = None) -> pd.DataFrame:
        try:
            if not end_date:
                end_date = datetime.now().strftime('%Y%m%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y%m%d')
            
            log.info(f"正在获取 {ts_code} 指数数据...")
            
            if self.ts_pro:
                df = self.ts_pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            elif HAS_AKSHARE:
                if ts_code == '000001.SH':
                    df = ak.stock_zh_index_daily(symbol="sh000001")
                else:
                    df = ak.stock_zh_index_daily(symbol="sz399001")
                df = df.rename(columns={
                    'date': 'trade_date',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'vol',
                    'amount': 'amount'
                })
                df['ts_code'] = ts_code
                df['pre_close'] = df['close'].shift(1)
                df['change'] = df['close'] - df['pre_close']
                df['pct_chg'] = df['change'] / df['pre_close'] * 100
                df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')
            else:
                raise ImportError("需要安装 AKShare 或 Tushare")
            
            log.info(f"获取到 {ts_code} {len(df)} 条指数数据")
            return df
        except Exception as e:
            log.error(f"获取指数数据失败: {e}")
            return pd.DataFrame()
    
    def save_index_daily(self, df: pd.DataFrame):
        if not HAS_AKSHARE and not HAS_TUSHARE:
            log.warning("数据存储功能需要 AKShare 或 Tushare")
            return
        
        if df.empty:
            return
        
        records = []
        for _, row in df.iterrows():
            record = {
                'ts_code': row['ts_code'],
                'trade_date': self._parse_date(row['trade_date']),
                'close': self._safe_float(row.get('close')),
                'open': self._safe_float(row.get('open')),
                'high': self._safe_float(row.get('high')),
                'low': self._safe_float(row.get('low')),
                'pre_close': self._safe_float(row.get('pre_close')),
                'change': self._safe_float(row.get('change')),
                'pct_chg': self._safe_float(row.get('pct_chg')),
                'vol': self._safe_float(row.get('vol')),
                'amount': self._safe_float(row.get('amount'))
            }
            records.append(record)
        
        db_manager.upsert_daily_data(IndexDaily, records, ['ts_code', 'trade_date'])
    
    def fetch_stock_daily_batch(self, ts_code_list: List[str], start_date: str = None, end_date: str = None, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票的日线数据（优先使用Tushare批量接口）
        
        Args:
            ts_code_list: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
        
        Returns:
            Dict[ts_code, DataFrame]
        """
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=300)).strftime('%Y%m%d')
        
        result = {}
        
        if use_cache:
            # 先选一支股票检查缓存状态
            check_ts_code = '000001.SZ'
            if check_ts_code not in ts_code_list and ts_code_list:
                check_ts_code = ts_code_list[0]
            
            print(f"\n检查缓存状态（使用 {check_ts_code} 验证）...")
            needs_update = self.check_cache_needs_update(check_ts_code, end_date)
            
            if not needs_update:
                print(f"  ✓ 缓存有效，从数据库加载所有 {len(ts_code_list)} 只股票数据...")
                batch_cache = self.load_stock_daily_batch_from_db(ts_code_list, None, end_date)
                
                total_load = len(ts_code_list)
                pb_load = create_progress_bar(total_load, '处理缓存数据')
                
                for i, ts_code in enumerate(ts_code_list):
                    if ts_code in batch_cache and not batch_cache[ts_code].empty:
                        result[ts_code] = batch_cache[ts_code]
                    pb_load.update(i + 1)
                
                pb_load.finish()
                log.info(f"从缓存批量加载 {len(result)} 只股票数据")
                print(f"  ✓ 成功加载 {len(result)}/{len(ts_code_list)} 只股票数据")
                return result
            else:
                print(f"  ✗ 发现有更新数据，需要批量获取所有股票...")
                ts_code_list = ts_code_list
        
        if not ts_code_list:
            return result
        
        log.info(f"需要获取 {len(ts_code_list)} 只股票的新数据")
        
        if self.ts_pro:
            try:
                print(f"\n[1/3] 尝试使用 Tushare 批量获取 (最快)...")
                log.info(f"使用Tushare批量获取 {len(ts_code_list)} 只股票数据...")
                
                all_new_records = []
                
                batch_size = 25
                total = len(ts_code_list)
                total_batches = (total + batch_size - 1) // batch_size
                pb = create_progress_bar(total_batches, '批量获取数据')
                
                request_count = 0
                start_time = datetime.now()
                
                for i in range(0, total, batch_size):
                    batch_codes = ts_code_list[i:i+batch_size]
                    codes_str = ','.join(batch_codes)
                    
                    request_count += 1
                    
                    df_batch = self.ts_pro.daily(ts_code=codes_str, start_date=start_date, end_date=end_date)
                    
                    # 检查 df_batch 是否为 None
                    if df_batch is not None and not df_batch.empty:
                        # 检查 df_batch 是否包含 'ts_code' 列
                        if 'ts_code' in df_batch.columns:
                            for ts_code in batch_codes:
                                df_stock = df_batch[df_batch['ts_code'] == ts_code].copy()
                                if not df_stock.empty:
                                    if 'vol' not in df_stock.columns and 'volume' in df_stock.columns:
                                        df_stock['vol'] = df_stock['volume']
                                    if 'vol' in df_stock.columns and 'volume' not in df_stock.columns:
                                        df_stock['volume'] = df_stock['vol']
                                    
                                    result[ts_code] = df_stock
                                    
                                    if use_cache:
                                        for _, row in df_stock.iterrows():
                                            # 检查 row 是否包含必要的列
                                            if 'ts_code' in row and row['ts_code'] is not None and 'trade_date' in row:
                                                record = {
                                                    'ts_code': row['ts_code'],
                                                    'trade_date': self._parse_date(row['trade_date']),
                                                    'open': self._safe_float(row.get('open')),
                                                    'high': self._safe_float(row.get('high')),
                                                    'low': self._safe_float(row.get('low')),
                                                    'close': self._safe_float(row.get('close')),
                                                    'pre_close': self._safe_float(row.get('pre_close')),
                                                    'change': self._safe_float(row.get('change')),
                                                    'pct_chg': self._safe_float(row.get('pct_chg')),
                                                    'vol': self._safe_float(row.get('vol')),
                                                    'amount': self._safe_float(row.get('amount'))
                                                }
                                                # 验证 record 是否有效
                                                if record['ts_code'] is not None and record['trade_date'] is not None:
                                                    all_new_records.append(record)
                                                else:
                                                    log.warning(f"跳过无效记录: ts_code={record['ts_code']}, trade_date={record['trade_date']}")
                        else:
                            log.warning(f"Tushare 返回的数据不包含 'ts_code' 列")
                    else:
                        log.warning(f"Tushare 返回空数据或 None for batch: {batch_codes[:3]}...")
                    
                    pb.update()
                    
                    if request_count % 45 == 0 and request_count > 0:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        if elapsed < 60:
                            wait_time = 60 - elapsed + 1
                            print(f"\n  ⏸️  已请求{request_count}次，暂停{wait_time:.0f}秒避免触发频率限制...")
                            import time
                            time.sleep(wait_time)
                            start_time = datetime.now()
                
                pb.finish()
                
                if use_cache and all_new_records:
                    if StockDaily is None:
                        log.warning("StockDaily 模型未加载，无法保存数据到数据库")
                        print("   ⚠️  数据模型未加载，跳过保存")
                    else:
                        print(f"\n正在批量保存 {len(all_new_records)} 条日线数据...")
                        pb_save = create_progress_bar(1, '保存数据')
                        db_manager.upsert_daily_data(StockDaily, all_new_records, ['ts_code', 'trade_date'])
                        pb_save.update()
                        pb_save.finish()
                        print(f"   ✓ 已保存 {len(all_new_records)} 条数据")
                
                log.info(f"Tushare批量获取完成，共获取 {len(result)} 只股票数据")
                return result
            
            except Exception as ts_error:
                print(f"\n⚠️  Tushare批量获取失败: {ts_error}")
                print(f"   直接尝试使用 AKShare...")
                log.warning(f"Tushare批量获取失败: {ts_error}，尝试AKShare")
        
        print(f"\n[3/3] 使用 AKShare 逐只获取 {len(ts_code_list)} 只股票数据...")
        pb = create_progress_bar(len(ts_code_list), '逐只获取(AKShare)')
        
        all_records = []
        for i, ts_code in enumerate(ts_code_list):
            try:
                # 直接实现 AKShare 获取逻辑，避免调用 fetch_stock_daily 产生过多日志
                symbol = ts_code.split('.')[0]
                df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                         start_date=start_date, end_date=end_date, adjust="qfq")
                df = df.rename(columns={
                    '日期': 'trade_date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'vol',
                    '成交额': 'amount',
                    '振幅': 'amplitude',
                    '涨跌幅': 'pct_chg',
                    '涨跌额': 'change',
                    '换手率': 'turnover'
                })
                if 'vol' in df.columns and 'volume' not in df.columns:
                    df['volume'] = df['vol']
                df['ts_code'] = ts_code
                df['pre_close'] = df['close'].shift(1)
                df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')
                
                if not df.empty:
                    result[ts_code] = df
                    if use_cache:
                        for _, row in df.iterrows():
                            # 检查 row 是否包含必要的列
                            if 'ts_code' in row and row['ts_code'] is not None and 'trade_date' in row:
                                record = {
                                    'ts_code': row['ts_code'],
                                    'trade_date': self._parse_date(row['trade_date']),
                                    'open': self._safe_float(row.get('open')),
                                    'high': self._safe_float(row.get('high')),
                                    'low': self._safe_float(row.get('low')),
                                    'close': self._safe_float(row.get('close')),
                                    'pre_close': self._safe_float(row.get('pre_close')),
                                    'change': self._safe_float(row.get('change')),
                                    'pct_chg': self._safe_float(row.get('pct_chg')),
                                    'vol': self._safe_float(row.get('vol')),
                                    'amount': self._safe_float(row.get('amount'))
                                }
                                # 验证 record 是否有效
                                if record['ts_code'] is not None and record['trade_date'] is not None:
                                    all_records.append(record)
                                else:
                                    log.warning(f"跳过无效记录: ts_code={record['ts_code']}, trade_date={record['trade_date']}")
            
            except Exception as e:
                # 只在严重错误时记录日志，避免干扰进度条
                if i % 100 == 0:
                    log.error(f"处理 {ts_code} 时出错: {e}")
            
            pb.update(i + 1)
        
        pb.finish()
        
        if use_cache and all_records:
            if StockDaily is None:
                log.warning("StockDaily 模型未加载，无法保存数据到数据库")
                print("   ⚠️  数据模型未加载，跳过保存")
            else:
                print(f"\n正在批量保存 {len(all_records)} 条日线数据...")
                pb_save = create_progress_bar(1, '保存数据')
                db_manager.upsert_daily_data(StockDaily, all_records, ['ts_code', 'trade_date'])
                pb_save.update()
                pb_save.finish()
                print(f"   ✓ 已保存 {len(all_records)} 条数据")
        
        log.info(f"AKShare逐只获取完成，共获取 {len(result)} 只股票数据")
        return result
    
    def download_all_stocks_daily(self, ts_code_list: List[str], start_date: str = None, end_date: str = None):
        if not HAS_AKSHARE and not HAS_TUSHARE:
            log.warning("数据下载功能需要 AKShare 或 Tushare")
            return
        
        log.info(f"开始下载 {len(ts_code_list)} 只股票的日线数据...")
        
        result = self.fetch_stock_daily_batch(ts_code_list, start_date, end_date, use_cache=True)
        
        log.info(f"所有股票日线数据下载完成，共处理 {len(result)} 只股票")
    
    def ensure_latest_data(self, ts_code_list: List[str]) -> bool:
        """
        确保所有股票都有最新的交易日数据
        
        Args:
            ts_code_list: 股票代码列表
            
        Returns:
            是否所有数据都是最新的
        """
        if not HAS_AKSHARE and not HAS_TUSHARE:
            return False
        
        log.info("正在检查所有股票数据是否为最新版本...")
        
        today_str = datetime.now().strftime('%Y%m%d')
        
        needs_update_list = []
        
        # 添加进度条
        pb = create_progress_bar(len(ts_code_list), '检查数据更新')
        
        for i, ts_code in enumerate(ts_code_list):
            if self.check_cache_needs_update(ts_code, today_str):
                needs_update_list.append(ts_code)
            pb.update(i + 1)
        
        pb.finish()
        
        if needs_update_list:
            log.info(f"发现 {len(needs_update_list)} 只股票需要更新数据")
            result = self.fetch_stock_daily_batch(needs_update_list, use_cache=True)
            log.info(f"已更新 {len(result)} 只股票的数据")
            return len(result) == len(needs_update_list)
        else:
            log.info("所有股票数据都是最新的")
            return True
    
    def _parse_date(self, date_val):
        if pd.isna(date_val) or date_val is None:
            return None
        if isinstance(date_val, datetime):
            return date_val.date()
        if isinstance(date_val, str):
            try:
                return datetime.strptime(date_val, '%Y%m%d').date()
            except:
                try:
                    return datetime.strptime(date_val, '%Y-%m-%d').date()
                except:
                    return None
        return None
    
    def _safe_float(self, val):
        if pd.isna(val) or val is None:
            return None
        try:
            return float(val)
        except:
            return None

data_fetcher = DataFetcher()
