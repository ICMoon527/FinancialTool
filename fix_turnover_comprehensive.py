
"""
全面修复换手率问题
1. 尝试直接从 AKShare API 获取正确的流通股本数据
2. 用流通股本重新计算并更新数据库中的所有换手率
"""
import logging
from datetime import date, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def fix_turnover_comprehensive():
    """
    全面修复换手率问题
    """
    print("="*70)
    print("全面修复换手率和流通股本")
    print("="*70)
    
    # 导入需要的模块
    from data_provider.base import DataFetcherManager
    from data_provider.akshare_fetcher import AkshareFetcher
    from src.services.turnover_service import TurnoverService
    from src.storage import DatabaseManager, StockDaily
    
    fetcher_manager = DataFetcherManager()
    turnover_service = TurnoverService()
    db = DatabaseManager.get_instance()
    
    # 获取所有有数据的股票
    print("\n1. 扫描数据库中的股票...")
    stock_codes = []
    try:
        with db.get_session() as session:
            from sqlalchemy import distinct
            result = session.execute(distinct(StockDaily.code)).all()
            stock_codes = [row[0] for row in result]
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return
    
    if not stock_codes:
        print("没有找到股票数据")
        return
    
    print(f"找到 {len(stock_codes)} 只股票")
    
    success_count = 0
    fail_count = 0
    
    for code in stock_codes[:10]:  # 先试10只
        print(f"\n{'='*60}")
        print(f"处理 {code}...")
        print(f"{'='*60}")
        
        try:
            # 1. 从API获取最新数据
            print(f"1.1 从API获取 {code} 的数据...")
            df, source = fetcher_manager.get_daily_data(code, days=700)
            
            if df is None or df.empty:
                print(f"   获取 {code} 数据失败")
                fail_count += 1
                continue
            
            # 2. 直接从 AKShare 获取流通股本
            print(f"1.2 尝试直接从 API 获取流通股本...")
            circulating_shares = None
            
            try:
                akshare_fetcher = AkshareFetcher()
                cs_df = akshare_fetcher.get_circulating_shares(code)
                
                if cs_df is not None and not cs_df.empty and 'circulating_shares' in cs_df.columns:
                    # 使用最新的流通股本
                    latest_cs = cs_df.sort_values('date').iloc[-1]['circulating_shares']
                    if latest_cs and latest_cs > 0:
                        circulating_shares = latest_cs
                        print(f"   从 AKShare API 获取到流通股本: {circulating_shares:.0f} 股")
            except Exception as e:
                print(f"   直接获取流通股本失败: {e}")
            
            # 3. 如果API获取不到，用新获取的数据重新估算
            if not circulating_shares or circulating_shares <= 0:
                print(f"1.3 用新获取的数据重新估算流通股本...")
                try:
                    basic_info = turnover_service.estimate_circulating_shares_from_akshare(df, code)
                    if basic_info and basic_info['circulating_shares'] > 0:
                        circulating_shares = basic_info['circulating_shares']
                        print(f"   重新估算成功: {circulating_shares:.0f} 股")
                        
                        # 保存到数据库
                        saved = turnover_service.save_stock_basic(basic_info)
                        if saved:
                            print(f"   流通股本信息已保存")
                except Exception as e:
                    print(f"   重新估算失败: {e}")
            
            # 4. 如果有了流通股本，重新计算换手率
            if circulating_shares and circulating_shares > 0:
                print(f"1.4 用流通股本重新计算换手率...")
                fill_count = turnover_service.fill_historical_turnover(code, circulating_shares)
                print(f"   更新了 {fill_count} 条换手率记录")
                
                # 验证一下新数据
                print(f"1.5 验证新数据...")
                end_date = date.today()
                start_date = end_date - timedelta(days=60)
                records = db.get_data_range(code, start_date, end_date)
                
                if records:
                    turnovers = [r.turnover_rate for r in records if r.turnover_rate is not None]
                    if turnovers:
                        print(f"   换手率范围: {min(turnovers):.6f} - {max(turnovers):.6f}")
                        if max(turnovers) > 0.01:
                            print(f"   ✓ 修复成功!")
                
                success_count += 1
            else:
                print(f"1.6 没有有效的流通股本数据，无法修复")
                fail_count += 1
        
        except Exception as e:
            print(f"   处理 {code} 时出错: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"修复完成:")
    print(f"成功: {success_count} 只")
    print(f"失败: {fail_count} 只")
    print(f"{'='*60}")

if __name__ == "__main__":
    fix_turnover_comprehensive()

