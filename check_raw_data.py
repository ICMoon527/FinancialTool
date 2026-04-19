
"""
检查数据源原始换手率格式
"""
import logging
import sys
import pandas as pd
from datetime import date, timedelta

logging.basicConfig(level=logging.INFO)

def check_efinance():
    """检查 efinance 原始数据格式"""
    print("="*60)
    print("检查 efinance 原始数据")
    print("="*60)
    
    try:
        import efinance as ef
        
        print("\n获取日线数据...")
        # 获取一只股票的日线数据
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        
        df = ef.stock.get_quote_history(
            '600397', 
            beg=str(start_date).replace('-', ''), 
            end=str(end_date).replace('-', '')
        )
        
        print(f"获取到 {len(df)} 条数据")
        print(f"\n列名: {list(df.columns)}")
        
        if '换手率' in df.columns:
            print(f"\n原始换手率数据:")
            for i, row in df.tail(10).iterrows():
                print(f"  {row['日期']}: {row['换手率']}")
            
            # 检查数据范围
            turnovers = df['换手率'].dropna()
            if len(turnovers) > 0:
                print(f"\n数据范围: min={turnovers.min()}, max={turnovers.max()}")
                print(f"平均值: {turnovers.mean()}")
                has_large = False
                has_small = False
                for t in turnovers:
                    if t > 1:
                        has_large = True
                    if t < 0.01:
                        has_small = True
                print(f"是否有大于1的值: {has_large}")
                print(f"是否有小于0.01的值: {has_small}")
                
    except Exception as e:
        print(f"efinance 检查失败: {e}")
        import traceback
        traceback.print_exc()

def check_akshare():
    """检查 akshare 原始数据格式"""
    print("\n" + "="*60)
    print("检查 akshare 原始数据")
    print("="*60)
    
    try:
        import akshare as ak
        
        print("\n获取日线数据...")
        # 获取一只股票的日线数据
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        
        df = ak.stock_zh_a_hist(
            symbol='600397',
            period='daily',
            start_date=str(start_date).replace('-', ''),
            end_date=str(end_date).replace('-', ''),
            adjust='qfq'
        )
        
        print(f"获取到 {len(df)} 条数据")
        print(f"\n列名: {list(df.columns)}")
        
        if '换手率' in df.columns:
            print(f"\n原始换手率数据:")
            for i, row in df.tail(10).iterrows():
                print(f"  {row['日期']}: {row['换手率']}")
            
            # 检查数据范围
            turnovers = df['换手率'].dropna()
            if len(turnovers) > 0:
                print(f"\n数据范围: min={turnovers.min()}, max={turnovers.max()}")
                print(f"平均值: {turnovers.mean()}")
                has_large = False
                has_small = False
                for t in turnovers:
                    if t > 1:
                        has_large = True
                    if t < 0.01:
                        has_small = True
                print(f"是否有大于1的值: {has_large}")
                print(f"是否有小于0.01的值: {has_small}")
                
    except Exception as e:
        print(f"akshare 检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_efinance()
    check_akshare()

