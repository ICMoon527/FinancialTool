
"""
调试换手率数据格式
"""
import logging
from datetime import date, timedelta
from src.storage import DatabaseManager, StockDaily
from sqlalchemy import select

logging.basicConfig(level=logging.INFO)

def main():
    db = DatabaseManager.get_instance()
    
    # 查找有换手率数据的股票
    with db.get_session() as session:
        # 查找有换手率数据的记录
        result = session.execute(
            select(StockDaily)
            .where(StockDaily.turnover_rate.isnot(None))
            .limit(1)
        ).fetchone()
        
        if not result:
            print("没有找到有换手率数据的记录")
            return
            
        stock_record = result[0]
        code = stock_record.code
        print(f"找到有换手率数据的股票代码: {code}")
        print(f"样本数据: 日期={stock_record.date}, 换手率={stock_record.turnover_rate}")
        
        # 获取这只股票最近的数据
        end_date = date.today()
        start_date = end_date - timedelta(days=120)
        
        data = db.get_data_range(code, start_date, end_date)
        
        if data:
            print(f"\n获取到 {len(data)} 条数据")
            print("\n换手率数据:")
            count = 0
            for i, d in enumerate(data):
                if d.turnover_rate is not None:
                    print(f"{count+1}. 日期: {d.date}, 换手率: {d.turnover_rate}")
                    count += 1
                if count >= 20:
                    break
                
            # 检查所有非空数据的范围
            all_turnovers = [d.turnover_rate for d in data if d.turnover_rate is not None]
            if all_turnovers:
                print(f"\n换手率范围: min={min(all_turnovers)}, max={max(all_turnovers)}")
                print(f"平均值: {sum(all_turnovers) / len(all_turnovers)}")
                
                # 计算数据中有多少在正常范围内
                count_normal = 0
                count_small = 0
                count_medium = 0
                count_large = 0
                for t in all_turnovers:
                    if 0 < t <= 0.1:
                        count_normal += 1
                    elif t < 0.001:
                        count_small += 1
                    elif t > 1:
                        count_large += 1
                    else:
                        count_medium += 1
                print(f"统计: 0-0.1范围: {count_normal}, 小于0.001: {count_small}, 大于1: {count_large}, 其他: {count_medium}")

if __name__ == "__main__":
    main()

