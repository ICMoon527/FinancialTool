# -*- coding: utf-8 -*-
"""强化学习分时做T交易信号系统"""

# ══════════════════════════════════════════════════════════════════
# Windows 原生崩溃防护（必须在任何 numpy/torch 导入之前设置）
# 背景：numpy(MKL) 与 torch 各自捆绑 libiomp5md.dll（OpenMP 运行时），
#       同进程共存时两套线程池并发互踩内存，会随机触发 0xC0000005 段错误
#       （崩点漂移：relu/linear/adam/numpy sample 随机出现）。
#       KMP_DUPLICATE_LIB_OK 只压制重复库致命报错，不能根治并发竞争。
# 根治：将 OMP/MKL 线程数强制为 1，所有 OpenMP 调用退化为串行，
#       彻底消除两套运行时共享线程池的竞争；KMP_BLOCKTIME=0 让并行区
#       结束后立即归还线程，进一步削弱线程驻留竞争。
# 注意：此包是 `python -m rl.*` 的第一个被导入模块，必须在最顶部执行，
#       否则 numpy 已随后续 import 加载，env 设置将失效。
# ══════════════════════════════════════════════════════════════════
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

from rl.config import RLConfig
from rl.environment import T0Environment
from rl.networks import DQNNetwork, DuelingDQNNetwork, create_dqn_network

__all__ = [
    "RLConfig",
    "T0Environment",
    "DQNNetwork",
    "DuelingDQNNetwork",
    "create_dqn_network",
]