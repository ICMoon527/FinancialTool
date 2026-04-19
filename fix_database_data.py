
"""
修复数据库中旧的错误换手率数据
此脚本将重新获取并覆盖现有数据
"""
import logging
from datetime import date, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def fix_stock_data(stock_code, force_refresh=False):
    """
    修复单个股票的数据
    
    Args:
        stock_code: 股票代码
        force_refresh: 是否强制刷新（即使有今天的数据也重新获取）
    """
    from data_provider.base import DataFetcherManager
    from src.storage import DatabaseManager
    
    logger = logging.getLogger('fix_data')
    
    fetcher_manager = DataFetcherManager()
    db = DatabaseManager.get_instance()
    
    logger.info(f"正在处理: {stock_code}")
    
    try:
        # 获取新数据（使用修复后的逻辑）
        df, source = fetcher_manager.get_daily_data(stock_code, days=700)
        
        if df is None or df.empty:
            logger.error(f"获取 {stock_code} 数据失败！")
            return False
        
        logger.info(f"获取到 {len(df)} 条数据，数据源: {source}")
        
        # 打印新数据的换手率信息
        if 'turnover_rate' in df.columns:
            turnovers = df['turnover_rate'].dropna()
            if len(turnovers) &gt; 0:
                logger.info(f"新数据 - 换手率范围: {turnovers.min():.6f} ~ {turnovers.max():.6f}")
                logger.info(f"新数据 - 换手率平均: {turnovers.mean():.6f}")
        
        # 保存到数据库（强制覆盖）
        saved_count = db.save_daily_data_bulk(df, stock_code, source)
        logger.info(f"成功保存/更新 {saved_count} 条数据到数据库")
        
        return True
        
    except Exception as e:
        logger.error(f"处理 {stock_code} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("="*70)
    print("数据库换手率数据修复工具")
    print("="*70)
    
    # 这里可以指定您需要修复的股票代码
    # 例如：stock_codes = ['600397', '000001', ...]
    # 或者先查询数据库中有哪些股票
    
    from src.storage import DatabaseManager
    
    db = DatabaseManager.get_instance()
    
    # 尝试获取数据库中已有的股票代码
    with db.get_session() as session:
        from sqlalchemy import distinct
        from src.storage import StockDaily
        
        result = session.execute(
            distinct(StockDaily.code)
        ).all()
        
        stock_codes = [row[0] for row in result]
    
    if not stock_codes:
        print("数据库中没有找到数据！")
        stock_codes = ['600397']  # 默认测试一个
    
    print(f"找到 {len(stock_codes)} 只股票需要修复")
    print(f"股票代码: {stock_codes[:10]}{'...' if len(stock_codes) &gt; 10 else ''}")
    
    # 逐个修复
    success_count = 0
    fail_count = 0
    
    for code in stock_codes:
        if fix_stock_data(code, force_refresh=True):
            success_count += 1
        else:
            fail_count += 1
        
        print()
    
    print("="*70)
    print(f"修复完成！")
    print(f"成功: {success_count} 只")
    print(f"失败: {fail_count} 只")
    print("="*70)

if __name__ == "__main__":
    main()

