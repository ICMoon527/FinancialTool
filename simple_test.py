#!/usr/bin/env python3
"""
简单测试脚本
"""
import sys
print("Hello, Python is working!")
print(f"Python version: {sys.version}")

try:
    import pandas as pd
    print("Pandas imported successfully")
except Exception as e:
    print(f"Pandas import failed: {e}")

try:
    from datetime import datetime
    print(f"Current time: {datetime.now()}")
except Exception as e:
    print(f"Datetime error: {e}")
