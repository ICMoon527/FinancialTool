"""
从 SQLite 数据库读取 600519（贵州茅台）日线数据，打印前5行和后5行，并保存为 CSV。
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text

DB_PATH = os.path.join("data", "stock_analysis.db")
CSV_PATH = "test_xma_data.csv"

engine = create_engine(f"sqlite:///{DB_PATH}")

with engine.connect() as conn:
    # 查询 stock_daily 表中 code='600519' 最近约300条数据，按 date 升序排列
    query = text("""
        SELECT date, open, high, low, close, volume
        FROM stock_daily
        WHERE code = '600519'
        ORDER BY date DESC
        LIMIT 300
    """)
    df = pd.read_sql_query(query, conn)
    # 按日期升序重新排列
    df = df.sort_values("date").reset_index(drop=True)

total = len(df)
print(f"共查询到 {total} 条数据")
if total > 0:
    print(f"日期范围：{df['date'].iloc[0]} ～ {df['date'].iloc[-1]}")

# 打印前5行
print("\n===== 前5行 =====")
print(df.head(5).to_string(index=False))

# 打印后5行
print("\n===== 后5行 =====")
print(df.tail(5).to_string(index=False))

# 保存为 CSV
df.to_csv(CSV_PATH, index=False)
print(f"\n数据已保存至 {CSV_PATH}")