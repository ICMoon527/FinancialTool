# -*- coding: utf-8 -*-
"""
多因子框架模块。
"""

from .base import (
    Factor,
    FactorConfig,
    FactorType,
    FactorDirection,
    FactorCombination,
    MomentumFactor,
    ValueFactor,
    VolatilityFactor,
    VolumeFactor,
    create_factor,
)

__all__ = [
    "Factor",
    "FactorConfig",
    "FactorType",
    "FactorDirection",
    "FactorCombination",
    "MomentumFactor",
    "ValueFactor",
    "VolatilityFactor",
    "VolumeFactor",
    "create_factor",
]
