import json
import os
from typing import Dict, Any, Optional
from logger import log


class StrategyConfig:
    
    def __init__(self, config_file: str = 'strategy_config.json'):
        self.config_file = config_file
        self.config = self._load_default_config()
        self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认策略配置"""
        return {
            'short': {
                'description': '短线策略（1-5个交易日）',
                'holding_days': 1,
                'target_return': 0.08,
                'stop_loss': 0.05,
                'entry_threshold': 70,
                'weights': {
                    'ma5_weight': 20.0,
                    'ma5_ma10_weight': 15.0,
                    'ma10_ma20_weight': 15.0,
                    'macd_positive_weight': 20.0,
                    'macd_cross_weight': 15.0,
                    'rsi_low_weight': 20.0,
                    'rsi_mid_weight': 15.0,
                    'kdj_weight': 20.0
                }
            },
            'medium': {
                'description': '中线策略（1-3个月）',
                'holding_days': 30,
                'target_return': 0.20,
                'stop_loss': 0.10,
                'entry_threshold': 70,
                'weights': {
                    'ma20_weight': 15.0,
                    'ma20_ma60_weight': 20.0,
                    'macd_weight': 20.0,
                    'trend_weight': 25.0,
                    'obv_weight': 20.0,
                    'volatility_weight': 20.0
                }
            },
            'long': {
                'description': '长线策略（6个月以上）',
                'holding_days': 180,
                'target_return': 0.50,
                'stop_loss': 0.15,
                'entry_threshold': 70,
                'weights': {
                    'ma20_weight': 15.0,
                    'ma20_ma60_weight': 20.0,
                    'ma60_ma120_weight': 15.0,
                    'trend_weight': 30.0,
                    'near_high_weight': 20.0,
                    'obv_weight': 20.0
                }
            },
            'optimization': {
                'objective': 'sharpe',
                'stock_count': 50,
                'data_days': 365,
                'early_stopping_rounds': 50,
                'parallel_workers': 4,
                'validation_method': 'rolling',
                'rolling_window_size': 120,
                'rolling_step_size': 30,
                'subsample_count': 3,
                'subsample_size': 180,
                'bootstrap_count': 5,
                'param_ranges': {
                    'short': {
                        'ma5_weight': {'min': 5.0, 'max': 30.0, 'step': 0.5},
                        'ma5_ma10_weight': {'min': 0.0, 'max': 25.0, 'step': 0.5},
                        'ma10_ma20_weight': {'min': 0.0, 'max': 25.0, 'step': 0.5},
                        'macd_positive_weight': {'min': 5.0, 'max': 35.0, 'step': 0.5},
                        'macd_cross_weight': {'min': 0.0, 'max': 25.0, 'step': 0.5},
                        'rsi_low_weight': {'min': 5.0, 'max': 30.0, 'step': 0.5},
                        'rsi_mid_weight': {'min': 0.0, 'max': 25.0, 'step': 0.5},
                        'kdj_weight': {'min': 5.0, 'max': 30.0, 'step': 0.5},
                        'entry_threshold': {'min': 50, 'max': 90, 'step': 5}
                    },
                    'medium': {
                        'ma20_weight': {'min': 0.0, 'max': 25.0, 'step': 0.5},
                        'ma20_ma60_weight': {'min': 5.0, 'max': 30.0, 'step': 0.5},
                        'macd_weight': {'min': 5.0, 'max': 30.0, 'step': 0.5},
                        'trend_weight': {'min': 10.0, 'max': 40.0, 'step': 0.5},
                        'obv_weight': {'min': 5.0, 'max': 30.0, 'step': 0.5},
                        'volatility_weight': {'min': 5.0, 'max': 30.0, 'step': 0.5},
                        'entry_threshold': {'min': 50, 'max': 90, 'step': 5}
                    },
                    'long': {
                        'ma20_weight': {'min': 0.0, 'max': 25.0, 'step': 0.5},
                        'ma20_ma60_weight': {'min': 5.0, 'max': 30.0, 'step': 0.5},
                        'ma60_ma120_weight': {'min': 5.0, 'max': 35.0, 'step': 0.5},
                        'trend_weight': {'min': 10.0, 'max': 50.0, 'step': 0.5},
                        'near_high_weight': {'min': 5.0, 'max': 30.0, 'step': 0.5},
                        'obv_weight': {'min': 5.0, 'max': 30.0, 'step': 0.5},
                        'entry_threshold': {'min': 50, 'max': 90, 'step': 5}
                    }
                }
            },
            'last_optimized': None,
            'optimization_results': {}
        }
    
    def _load_config(self):
        """从文件加载配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                self._merge_config(loaded_config)
                log.info(f"策略配置已从 {self.config_file} 加载")
            except Exception as e:
                log.warning(f"加载策略配置失败: {e}，使用默认配置")
    
    def _merge_config(self, loaded_config: Dict):
        """合并加载的配置到默认配置"""
        for key in loaded_config:
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(loaded_config[key], dict):
                    self.config[key].update(loaded_config[key])
                else:
                    self.config[key] = loaded_config[key]
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            log.info(f"策略配置已保存到 {self.config_file}")
        except Exception as e:
            log.error(f"保存策略配置失败: {e}")
    
    def get_config(self, horizon: str) -> Optional[Dict]:
        """获取指定周期的策略配置"""
        return self.config.get(horizon)
    
    def get_weights(self, horizon: str) -> Dict:
        """获取指定周期的权重配置"""
        horizon_config = self.config.get(horizon, {})
        return horizon_config.get('weights', {})
    
    def update_weights(self, horizon: str, weights: Dict):
        """更新指定周期的权重配置"""
        if horizon in self.config:
            self.config[horizon]['weights'].update(weights)
            self.save_config()
    
    def update_entry_threshold(self, horizon: str, threshold: int):
        """更新入场阈值"""
        if horizon in self.config:
            self.config[horizon]['entry_threshold'] = threshold
            self.save_config()
    
    def save_optimization_results(self, results: Dict):
        """保存优化结果"""
        from datetime import datetime
        self.config['optimization_results'] = results
        self.config['last_optimized'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for horizon in ['short', 'medium', 'long']:
            if horizon in results and 'best_params' in results[horizon]:
                best_params = results[horizon]['best_params']
                if best_params:
                    weights = {k: v for k, v in best_params.items() if k != 'entry_threshold'}
                    self.update_weights(horizon, weights)
                    if 'entry_threshold' in best_params:
                        self.update_entry_threshold(horizon, best_params['entry_threshold'])
        
        self.save_config()
        log.info("优化结果已保存并更新策略配置")
    
    def get_param_ranges(self, horizon: str) -> Dict:
        """获取参数优化范围"""
        return self.config.get('optimization', {}).get('param_ranges', {}).get(horizon, {})
    
    def get_early_stopping_rounds(self) -> int:
        """获取早停轮数"""
        return self.config.get('optimization', {}).get('early_stopping_rounds', 50)
    
    def get_parallel_workers(self) -> int:
        """获取并行工作数"""
        return self.config.get('optimization', {}).get('parallel_workers', 4)
    
    def get_validation_method(self) -> str:
        """获取验证方法"""
        return self.config.get('optimization', {}).get('validation_method', 'rolling')
    
    def get_rolling_window_size(self) -> int:
        """获取滚动窗口大小"""
        return self.config.get('optimization', {}).get('rolling_window_size', 120)
    
    def get_rolling_step_size(self) -> int:
        """获取滚动窗口步长"""
        return self.config.get('optimization', {}).get('rolling_step_size', 30)
    
    def get_subsample_count(self) -> int:
        """获取子采样数量"""
        return self.config.get('optimization', {}).get('subsample_count', 3)
    
    def get_subsample_size(self) -> int:
        """获取子采样大小"""
        return self.config.get('optimization', {}).get('subsample_size', 180)
    
    def get_bootstrap_count(self) -> int:
        """获取Bootstrap数量"""
        return self.config.get('optimization', {}).get('bootstrap_count', 5)
    
    def get_current_params(self, horizon: str) -> Dict:
        """获取当前配置的参数"""
        horizon_config = self.config.get(horizon, {})
        weights = horizon_config.get('weights', {})
        entry_threshold = horizon_config.get('entry_threshold', 70)
        
        params = weights.copy()
        params['entry_threshold'] = entry_threshold
        return params
    
    def reset_to_default(self):
        """重置为默认配置"""
        self.config = self._load_default_config()
        self.save_config()
        log.info("策略配置已重置为默认值")


strategy_config = StrategyConfig()
