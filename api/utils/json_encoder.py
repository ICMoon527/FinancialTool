# -*- coding: utf-8 -*-
"""
自定义 JSON 编码器，支持 numpy 数据类型序列化
"""
import json
from typing import Any
import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """
    支持 numpy 数据类型的 JSON 编码器
    """
    def default(self, obj: Any) -> Any:
        # numpy 布尔值
        if isinstance(obj, np.bool_):
            return bool(obj)
        # numpy 整数
        if isinstance(obj, np.integer):
            return int(obj)
        # numpy 浮点数
        if isinstance(obj, np.floating):
            return float(obj)
        # numpy 数组
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # 其他类型，使用默认处理
        return super().default(obj)


def jsonable_encoder_with_numpy(obj: Any) -> Any:
    """
    将对象转换为可 JSON 序列化的格式，支持 numpy 类型
    
    Args:
        obj: 需要转换的对象
    
    Returns:
        可 JSON 序列化的对象
    """
    if isinstance(obj, dict):
        return {k: jsonable_encoder_with_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable_encoder_with_numpy(item) for item in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
