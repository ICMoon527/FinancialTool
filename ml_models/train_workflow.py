"""机器学习模型训练工作流"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from ml_models.config import default_config, ModelConfig
from ml_models.features import FeatureExtractor
from ml_models.models import (
    BaseModel,
    XGBoostModel,
    LightGBMModel,
    LSTMModel,
    GRUModel,
    EnsembleModel
)
from logger import log
from config import user_config

try:
    from data.database import db_manager, HAS_SQLALCHEMY
except ImportError:
    HAS_SQLALCHEMY = False
    db_manager = None

from data.sample_data import sample_data_generator


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: ModelConfig = None):
        """初始化模型训练器"""
        self.config = config or default_config
        self.feature_extractor = FeatureExtractor(self.config)
        
        # 创建保存目录
        self.model_save_dir = Path(self.config.model_save_dir)
        self.report_save_dir = Path(self.config.model_save_dir).parent / 'reports'
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.report_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型映射
        self.model_classes = {
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'lstm': LSTMModel,
            'gru': GRUModel,
            'ensemble': EnsembleModel
        }
        
        # 训练结果存储
        self.training_results = {}
    
    def load_data(self, use_sample: bool = True, max_stocks: int = 20) -> Dict[str, pd.DataFrame]:
        """
        加载股票数据
        
        Args:
            use_sample: 是否使用样本数据
            max_stocks: 最大股票数量
        
        Returns:
            股票数据字典
        """
        log.info("正在加载股票数据...")
        
        if use_sample or not HAS_SQLALCHEMY or db_manager is None:
            log.info("使用样本数据进行训练")
            return sample_data_generator.generate_sample_stock_data(num_stocks=max_stocks, days=365)
        
        try:
            log.info("从数据库加载真实数据...")
            stock_data = db_manager.load_all_stock_data(max_stocks=max_stocks)
            
            if not stock_data:
                log.warning("数据库中没有数据，使用样本数据")
                return sample_data_generator.generate_sample_stock_data(num_stocks=max_stocks, days=365)
            
            log.info(f"成功加载 {len(stock_data)} 支股票数据")
            return stock_data
        except Exception as e:
            log.error(f"从数据库加载数据失败: {e}")
            log.info("使用样本数据进行训练")
            return sample_data_generator.generate_sample_stock_data(num_stocks=max_stocks, days=365)
    
    def prepare_training_data(self, stock_data: Dict[str, pd.DataFrame], target_days: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备训练数据
        
        Args:
            stock_data: 股票数据字典
            target_days: 预测目标天数
        
        Returns:
            (X, y) 特征矩阵和目标向量
        """
        log.info(f"正在准备训练数据，预测目标: {target_days}天...")
        
        all_X = []
        all_y = []
        
        for ts_code, df in stock_data.items():
            if df.empty or len(df) < target_days + 30:
                continue
            
            try:
                X, y = self.feature_extractor.prepare_training_data(df, target_days)
                if X is not None and y is not None and len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
            except Exception as e:
                log.warning(f"处理股票 {ts_code} 时出错: {e}")
                continue
        
        if not all_X:
            raise ValueError("没有可用的训练数据")
        
        # 合并数据
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        
        # 移除无穷值和NaN
        mask = np.isfinite(y_combined)
        X_combined = X_combined[mask]
        y_combined = y_combined[mask]
        
        log.info(f"训练数据准备完成: {len(X_combined)} 样本, {X_combined.shape[1]} 特征")
        
        return X_combined, y_combined
    
    def train_single_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, 
                          do_tuning: bool = False) -> Tuple[object, Dict]:
        """
        训练单个模型
        
        Args:
            model_name: 模型名称
            X: 特征矩阵
            y: 目标向量
            do_tuning: 是否进行超参数调优
        
        Returns:
            (model, metrics) 训练好的模型和评估指标
        """
        log.info(f"正在训练 {model_name} 模型...")
        
        try:
            model_class = self.model_classes[model_name]
            model = model_class(self.config)
            
            # 超参数调优
            if do_tuning and hasattr(model, 'hyperparameter_tuning') and model_name in ['xgboost', 'lightgbm']:
                log.info(f"正在进行 {model_name} 超参数调优...")
                tuning_result = model.hyperparameter_tuning(X, y)
                log.info(f"{model_name} 最佳参数: {tuning_result.get('best_params', {})}")
            
            # 训练模型
            rmse = model.train(X, y)
            
            # 评估模型
            metrics = model.evaluate(X, y)
            metrics['rmse_train'] = rmse
            
            log.info(f"{model_name} 训练完成 - RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.4f}, 方向准确率: {metrics['direction_accuracy']:.4f}")
            
            return model, metrics
            
        except Exception as e:
            log.error(f"训练 {model_name} 模型失败: {e}")
            import traceback
            traceback.print_exc()
            return None, {}
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series, target_days: int,
                        model_types: List[str] = None, do_tuning: bool = False) -> Dict:
        """
        训练所有模型
        
        Args:
            X: 特征矩阵
            y: 目标向量
            target_days: 预测目标天数
            model_types: 要训练的模型类型列表
            do_tuning: 是否进行超参数调优
        
        Returns:
            训练结果字典
        """
        if model_types is None:
            model_types = ['xgboost', 'lightgbm', 'lstm', 'gru', 'ensemble']
        
        results = {
            'target_days': target_days,
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for model_name in model_types:
            model, metrics = self.train_single_model(model_name, X, y, do_tuning)
            
            if model is not None:
                # 保存模型
                model_path = self._save_model(model, model_name, target_days)
                
                results['models'][model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'model_path': model_path
                }
        
        self.training_results[target_days] = results
        return results
    
    def _save_model(self, model: object, model_name: str, target_days: int) -> str:
        """
        保存模型到文件
        
        Args:
            model: 模型对象
            model_name: 模型名称
            target_days: 预测目标天数
        
        Returns:
            模型保存路径
        """
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"{model_name}_{target_days}d_{timestamp}"
        
        if model_name in ['lstm', 'gru']:
            # PyTorch模型
            model_path = self.model_save_dir / f"{filename}.pt"
            model.save(str(model_path))
        elif model_name == 'xgboost':
            # XGBoost模型
            model_path = self.model_save_dir / f"{filename}.json"
            model.save(str(model_path))
        else:
            # 其他模型用pickle
            model_path = self.model_save_dir / f"{filename}.pkl"
            import joblib
            joblib.dump(model, str(model_path))
        
        # 同时保存最新版本（不带时间戳）
        latest_path = self.model_save_dir / f"{model_name}_{target_days}d_latest"
        if model_name in ['lstm', 'gru']:
            model.save(str(latest_path) + '.pt')
        elif model_name == 'xgboost':
            model.save(str(latest_path) + '.json')
        else:
            import joblib
            joblib.dump(model, str(latest_path) + '.pkl')
        
        log.info(f"模型已保存: {model_path}")
        return str(model_path)
    
    def load_model(self, model_name: str, target_days: int, use_latest: bool = True) -> Optional[object]:
        """
        加载已保存的模型
        
        Args:
            model_name: 模型名称
            target_days: 预测目标天数
            use_latest: 是否使用最新版本
        
        Returns:
            加载的模型对象
        """
        if use_latest:
            base_path = self.model_save_dir / f"{model_name}_{target_days}d_latest"
        else:
            # 查找最新的带时间戳的文件
            import glob
            pattern = str(self.model_save_dir / f"{model_name}_{target_days}d_*.")
            files = glob.glob(pattern + '*')
            if not files:
                log.error(f"找不到模型文件: {model_name}_{target_days}d")
                return None
            base_path = Path(max(files, key=os.path.getctime)).rsplit('.', 1)[0]
        
        try:
            model_class = self.model_classes[model_name]
            model = model_class(self.config)
            
            if model_name in ['lstm', 'gru']:
                model.load(str(base_path) + '.pt')
            elif model_name == 'xgboost':
                model.load(str(base_path) + '.json')
            else:
                import joblib
                model = joblib.load(str(base_path) + '.pkl')
            
            log.info(f"模型已加载: {base_path}")
            return model
            
        except Exception as e:
            log.error(f"加载模型失败: {e}")
            return None
    
    def generate_report(self, results: Dict, target_days: int) -> str:
        """
        生成训练报告
        
        Args:
            results: 训练结果
            target_days: 预测目标天数
        
        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.report_save_dir / f"training_report_{target_days}d_{timestamp}.json"
        
        # 准备报告数据
        report_data = {
            'target_days': target_days,
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'models': {}
        }
        
        # 收集各模型指标
        all_metrics = []
        for model_name, result in results['models'].items():
            metrics = result['metrics']
            report_data['models'][model_name] = {
                'metrics': metrics,
                'model_path': result.get('model_path', '')
            }
            all_metrics.append({
                'model': model_name,
                **metrics
            })
        
        # 生成摘要
        if all_metrics:
            df_metrics = pd.DataFrame(all_metrics)
            report_data['summary'] = {
                'best_rmse': df_metrics.loc[df_metrics['rmse'].idxmin()]['model'],
                'best_r2': df_metrics.loc[df_metrics['r2'].idxmax()]['model'],
                'best_direction': df_metrics.loc[df_metrics['direction_accuracy'].idxmax()]['model'],
                'all_metrics': df_metrics.to_dict('records')
            }
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        # 同时保存为最新报告
        latest_report_path = self.report_save_dir / f"training_report_{target_days}d_latest.json"
        with open(latest_report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        log.info(f"训练报告已生成: {report_path}")
        
        # 打印报告摘要
        self._print_report_summary(report_data)
        
        return str(report_path)
    
    def _print_report_summary(self, report_data: Dict):
        """打印报告摘要"""
        print("\n" + "="*80)
        print("【模型训练报告摘要】")
        print("="*80)
        
        print(f"\n预测目标: {report_data['target_days']}天")
        print(f"训练时间: {report_data['timestamp']}")
        
        summary = report_data.get('summary', {})
        if summary:
            print(f"\n最佳模型:")
            print(f"  - 最小RMSE: {summary.get('best_rmse', 'N/A')}")
            print(f"  - 最高R²: {summary.get('best_r2', 'N/A')}")
            print(f"  - 最高方向准确率: {summary.get('best_direction', 'N/A')}")
        
        print("\n各模型性能:")
        for model_name, model_data in report_data['models'].items():
            metrics = model_data['metrics']
            print(f"\n  {model_name.upper()}:")
            print(f"    RMSE: {metrics.get('rmse', 0):.6f}")
            print(f"    R²: {metrics.get('r2', 0):.4f}")
            print(f"    方向准确率: {metrics.get('direction_accuracy', 0):.4f}")
        
        print("\n" + "="*80)
    
    def run(self, target_days_list: List[int] = None, model_types: List[str] = None,
           use_sample: bool = True, max_stocks: int = None, do_tuning: bool = False):
        """
        运行完整训练工作流
        
        Args:
            target_days_list: 预测目标天数列表
            model_types: 要训练的模型类型
            use_sample: 是否使用样本数据
            max_stocks: 最大股票数量（默认从user_config读取）
            do_tuning: 是否进行超参数调优
        """
        if target_days_list is None:
            target_days_list = [1, 5, 20]
        
        if max_stocks is None:
            max_stocks = user_config.stock_count
        
        log.info("="*80)
        log.info("开始机器学习模型训练工作流")
        log.info("="*80)
        
        # 加载数据
        stock_data = self.load_data(use_sample=use_sample, max_stocks=max_stocks)
        
        # 对每个预测目标进行训练
        all_reports = {}
        
        for target_days in target_days_list:
            log.info(f"\n{'='*80}")
            log.info(f"训练预测目标: {target_days}天")
            log.info(f"{'='*80}")
            
            try:
                # 准备训练数据
                X, y = self.prepare_training_data(stock_data, target_days)
                
                # 训练模型
                results = self.train_all_models(X, y, target_days, model_types, do_tuning)
                
                # 生成报告
                report_path = self.generate_report(results, target_days)
                all_reports[target_days] = report_path
                
            except Exception as e:
                log.error(f"训练 {target_days} 天模型时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        log.info("\n" + "="*80)
        log.info("训练工作流完成!")
        log.info("="*80)
        
        return all_reports


def main():
    """主函数"""
    import argparse
    
    default_max_stocks = user_config.stock_count
    
    parser = argparse.ArgumentParser(description='机器学习模型训练工作流')
    parser.add_argument('--target-days', type=int, nargs='+', default=[1, 5, 20],
                       help='预测目标天数列表 (默认: 1 5 20)')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['xgboost', 'lightgbm', 'lstm', 'gru', 'ensemble'],
                       help='要训练的模型类型 (默认: xgboost lightgbm lstm gru ensemble)')
    parser.add_argument('--use-sample', action='store_true', default=True,
                       help='使用样本数据 (默认: True)')
    parser.add_argument('--no-sample', action='store_false', dest='use_sample',
                       help='不使用样本数据，尝试从数据库加载')
    parser.add_argument('--max-stocks', type=int, default=default_max_stocks,
                       help=f'最大股票数量 (默认: {default_max_stocks}, 与user_config同步)')
    parser.add_argument('--tuning', action='store_true',
                       help='进行超参数调优 (默认: False)')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = ModelTrainer()
    
    # 运行训练
    trainer.run(
        target_days_list=args.target_days,
        model_types=args.models,
        use_sample=args.use_sample,
        max_stocks=args.max_stocks,
        do_tuning=args.tuning
    )


if __name__ == '__main__':
    main()
