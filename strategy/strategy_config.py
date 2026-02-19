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
                    'ma5_weight': 20,
                    'ma5_ma10_weight': 15,
                    'ma10_ma20_weight': 15,
                    'macd_positive_weight': 20,
                    'macd_cross_weight': 15,
                    'rsi_low_weight': 20,
                    'rsi_mid_weight': 15,
                    'kdj_weight': 20
                }
            },
            'medium': {
                'description': '中线策略（1-3个月）',
                'holding_days': 30,
                'target_return': 0.20,
                'stop_loss': 0.10,
                'entry_threshold': 70,
                'weights': {
                    'ma20_weight': 15,
                    'ma20_ma60_weight': 20,
                    'macd_weight': 20,
                    'trend_weight': 25,
                    'obv_weight': 20,
                    'volatility_weight': 20
                }
            },
            'long': {
                'description': '长线策略（6个月以上）',
                'holding_days': 180,
                'target_return': 0.50,
                'stop_loss': 0.15,
                'entry_threshold': 70,
                'weights': {
                    'ma20_weight': 15,
                    'ma20_ma60_weight': 20,
                    'ma60_ma120_weight': 15,
                    'trend_weight': 30,
                    'near_high_weight': 20,
                    'obv_weight': 20
                }
            },
            'optimization': {
                'objective': 'sharpe',
                'stock_count': 50,
                'data_days': 365,
                'param_ranges': {
                    'short': {
                        'ma5_weight': [10, 15, 20],
                        'ma5_ma10_weight': [5, 10, 15],
                        'ma10_ma20_weight': [5, 10, 15],
                        'macd_positive_weight': [15, 20, 25],
                        'macd_cross_weight': [5, 10, 15],
                        'rsi_low_weight': [10, 15, 20],
                        'rsi_mid_weight': [5, 10, 15],
                        'kdj_weight': [10, 15, 20],
                        'entry_threshold': [60, 70, 80]
                    },
                    'medium': {
                        'ma20_weight': [5, 10, 15],
                        'ma20_ma60_weight': [10, 15, 20],
                        'macd_weight': [10, 15, 20],
                        'trend_weight': [15, 20, 25],
                        'obv_weight': [10, 15, 20],
                        'volatility_weight': [10, 15, 20],
                        'entry_threshold': [60, 70, 80]
                    },
                    'long': {
                        'ma20_weight': [5, 10, 15],
                        'ma20_ma60_weight': [10, 15, 20],
                        'ma60_ma120_weight': [15, 20, 25],
                        'trend_weight': [20, 25, 30],
                        'near_high_weight': [10, 15, 20],
                        'obv_weight': [10, 15, 20],
                        'entry_threshold': [60, 70, 80]
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
    
    def reset_to_default(self):
        """重置为默认配置"""
        self.config = self._load_default_config()
        self.save_config()
        log.info("策略配置已重置为默认值")


strategy_config = StrategyConfig()
