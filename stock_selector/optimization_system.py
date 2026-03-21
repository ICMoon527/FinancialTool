# -*- coding: utf-8 -*-
"""
Strategy Optimization System Data Structures.

Core data structures for the strategy continuous optimization and iteration system.
"""

"""
python -m stock_selector.optimization_system walk-forward --strategies all --update-config

# List available strategies (Windows/Linux/Mac)
python -m stock_selector.optimization_system list

# Run full optimization (single line - works on all platforms)
python -m stock_selector.optimization_system optimize --strategies all --start-date 2025-01-01 --max-iterations 50 --save records.json

# Load and display records
python -m stock_selector.optimization_system load --file records.json
"""


import json
import logging
import concurrent.futures
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import yaml

from src.repositories.stock_repo import StockRepository
from src.repositories.backtest_repo import BacktestRepository
from src.storage import DatabaseManager, StockDaily, BacktestResult, SectorDaily
from src.core.backtest_engine import BacktestEngine, EvaluationConfig
from src.services.backtest_service import BacktestService
from sqlalchemy import and_, select, desc, func
from stock_selector.base import StockCandidate, StrategyMatch
from stock_selector.service import StockSelectorService
from stock_selector.config import get_config
from stock_selector.stock_pool import get_all_stock_codes
import uuid


logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objective types."""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    CALMAR_RATIO = "calmar_ratio"
    RANK_IC = "rank_ic"


class OptimizationMethod(Enum):
    """Optimization method types."""
    BAYESIAN = "bayesian"
    GRID_SEARCH = "grid_search"


class ProgressBar:
    """
    Simple progress bar for console display.
    
    Features:
    - Dynamic update without multiple lines
    - Shows completed count, total count, and percentage
    - Support different quantities of items
    - Minimal performance impact
    """
    
    def __init__(self, total: int, description: str = "Processing", width: int = 50):
        """
        Initialize progress bar.
        
        Args:
            total: Total number of items to process
            description: Description text to show before the bar
            width: Width of the progress bar in characters
        """
        self.total = total
        self.description = description
        self.width = width
        self.current = 0
        self.start_time = datetime.now()
    
    def update(self, current: int) -> None:
        """
        Update progress bar with current progress.
        
        Args:
            current: Current number of completed items
        """
        self.current = current
        self._render()
    
    def increment(self) -> None:
        """Increment progress by one."""
        self.current += 1
        self._render()
    
    def _render(self) -> None:
        """Render the progress bar to console."""
        if self.total == 0:
            return
        
        percentage = min(100, (self.current / self.total) * 100)
        filled = int(self.width * self.current / self.total)
        bar = '█' * filled + '░' * (self.width - filled)
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
        else:
            eta = 0
        
        sys.stdout.write(f'\r{self.description}: [{bar}] {self.current}/{self.total} ({percentage:.1f}%) - ETA: {eta:.1f}s')
        sys.stdout.flush()
    
    def finish(self, message: str = "Done!") -> None:
        """
        Finish the progress bar and show completion message.
        
        Args:
            message: Completion message to display
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()
        sys.stdout.write(f'\r{self.description}: [{"█" * self.width}] {self.total}/{self.total} (100.0%) - Elapsed: {elapsed:.1f}s\n')
        sys.stdout.write(f'{message}\n')
        sys.stdout.flush()


class OptimizationDirection(Enum):
    """Optimization direction enum."""
    INCREASE_SCORE_MULTIPLIER = "increase_score_multiplier"
    DECREASE_SCORE_MULTIPLIER = "decrease_score_multiplier"
    ADJUST_PARAMETERS = "adjust_parameters"
    NO_CHANGE = "no_change"


@dataclass
class StrategyParamsSnapshot:
    """Strategy parameter snapshot storing strategy configuration."""
    
    strategy_id: str
    strategy_name: str
    score_multiplier: float = 1.0
    max_raw_score: float = 100.0
    enabled: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Ensure score_multiplier is within valid bounds after initialization."""
        min_val = 0.1
        max_val = 5.0
        self.score_multiplier = max(min_val, min(max_val, self.score_multiplier))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyParamsSnapshot':
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StrategyParamsSnapshot':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class StockSelectionRecord:
    """Single stock selection record."""
    
    code: str
    name: str
    buy_price: float
    match_score: float
    strategy_matches: List[Dict[str, Any]] = field(default_factory=list)
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StockSelectionRecord':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class StockSelectionResult:
    """Stock selection result storing stock list and details."""
    
    selection_date: datetime
    selected_stocks: List[StockSelectionRecord] = field(default_factory=list)
    total_candidates: int = 0
    min_match_score: float = 0.0
    top_n: Optional[int] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['selection_date'] = self.selection_date.isoformat()
        data['selected_stocks'] = [stock.to_dict() for stock in self.selected_stocks]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StockSelectionResult':
        """Create from dictionary."""
        data = data.copy()
        data['selection_date'] = datetime.fromisoformat(data['selection_date'])
        data['selected_stocks'] = [StockSelectionRecord.from_dict(stock) for stock in data['selected_stocks']]
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StockSelectionResult':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class StockBacktestDetail:
    """Detailed backtest result for a single stock."""
    
    code: str
    name: str
    stock_return_pct: float
    simulated_return_pct: Optional[float] = None
    outcome: Optional[str] = None
    strategy_matches: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BacktestResultSnapshot:
    """Backtest result snapshot storing backtest metrics."""
    
    backtest_id: str
    start_date: datetime
    end_date: datetime
    total_return_pct: float = 0.0
    win_rate_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    direction_accuracy_pct: Optional[float] = None
    avg_stock_return_pct: Optional[float] = None
    avg_simulated_return_pct: Optional[float] = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    neutral_trades: int = 0
    stop_loss_trigger_rate_pct: Optional[float] = None
    take_profit_trigger_rate_pct: Optional[float] = None
    avg_days_to_first_hit: Optional[float] = None
    eval_window_days: int = 20
    engine_version: str = "v1"
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    stock_details: List[StockBacktestDetail] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['start_date'] = self.start_date.isoformat()
        data['end_date'] = self.end_date.isoformat()
        data['created_at'] = self.created_at.isoformat()
        data['stock_details'] = [asdict(detail) for detail in self.stock_details]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestResultSnapshot':
        """Create from dictionary."""
        data = data.copy()
        data['start_date'] = datetime.fromisoformat(data['start_date'])
        data['end_date'] = datetime.fromisoformat(data['end_date'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        stock_details_data = data.pop('stock_details', [])
        stock_details = []
        for detail_data in stock_details_data:
            stock_details.append(StockBacktestDetail(**detail_data))
        
        instance = cls(**data)
        instance.stock_details = stock_details
        return instance
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BacktestResultSnapshot':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class IterationRecord:
    """Iteration record storing complete information for a single iteration."""
    
    iteration_id: str
    iteration_number: int
    timestamp: datetime
    strategy_params: Dict[str, StrategyParamsSnapshot]  # Changed to dict for multiple strategies
    stock_selection_result: StockSelectionResult
    backtest_result: BacktestResultSnapshot
    optimization_direction: OptimizationDirection
    optimization_reason: Optional[str] = None
    parent_iteration_id: Optional[str] = None
    performance_improvement_pct: Optional[float] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['strategy_params'] = {k: v.to_dict() for k, v in self.strategy_params.items()}
        data['stock_selection_result'] = self.stock_selection_result.to_dict()
        data['backtest_result'] = self.backtest_result.to_dict()
        data['optimization_direction'] = self.optimization_direction.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IterationRecord':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['strategy_params'] = {
            k: StrategyParamsSnapshot.from_dict(v) 
            for k, v in data['strategy_params'].items()
        }
        data['stock_selection_result'] = StockSelectionResult.from_dict(data['stock_selection_result'])
        data['backtest_result'] = BacktestResultSnapshot.from_dict(data['backtest_result'])
        data['optimization_direction'] = OptimizationDirection(data['optimization_direction'])
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'IterationRecord':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class OptimizationConfig:
    """Optimization configuration storing optimization parameters."""
    
    max_iterations: int = 100
    convergence_threshold_pct: float = 0.1
    adjustment_step_pct: float = 5.0
    min_score_multiplier: float = 0.1
    max_score_multiplier: float = 5.0
    target_return_pct: Optional[float] = None
    target_win_rate_pct: Optional[float] = None
    max_drawdown_limit_pct: Optional[float] = None
    backtest_window_days: int = 20
    rebalance_frequency_days: int = 5
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    random_seed: Optional[int] = None
    enable_sector_data: bool = False  # Whether to fetch and use sector data in historical mode
    # Optimization method and objective
    optimization_method: OptimizationMethod = OptimizationMethod.BAYESIAN
    optimization_objective: OptimizationObjective = OptimizationObjective.TOTAL_RETURN
    # Bayesian optimization parameters
    bayesian_n_calls: int = 30
    bayesian_n_initial_points: int = 5
    # Grid search parameters
    grid_search_step: float = 0.1
    normalize_weights: bool = True
    # Walk-Forward Analysis (WFA) parameters
    wfa_enabled: bool = False
    wfa_training_window_days: int = 60
    wfa_testing_window_days: int = 20
    wfa_step_days: int = 20
    wfa_num_windows: int = 5
    wfa_start_date: Optional[date] = field(default_factory=lambda: date(2021, 1, 1))
    extra_config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['optimization_method'] = self.optimization_method.value
        data['optimization_objective'] = self.optimization_objective.value
        if self.wfa_start_date:
            data['wfa_start_date'] = self.wfa_start_date.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        # Parse optimization method
        if 'optimization_method' in data and isinstance(data['optimization_method'], str):
            try:
                data['optimization_method'] = OptimizationMethod(data['optimization_method'])
            except ValueError:
                del data['optimization_method']
        
        # Parse optimization objective
        if 'optimization_objective' in data and isinstance(data['optimization_objective'], str):
            try:
                data['optimization_objective'] = OptimizationObjective(data['optimization_objective'])
            except ValueError:
                del data['optimization_objective']
        
        # Parse wfa_start_date
        if 'wfa_start_date' in data and data['wfa_start_date'] and isinstance(data['wfa_start_date'], str):
            try:
                data['wfa_start_date'] = date.fromisoformat(data['wfa_start_date'])
            except ValueError:
                del data['wfa_start_date']
        
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'OptimizationConfig':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def from_yaml(cls, config_path: Optional[Path] = None) -> 'OptimizationConfig':
        """
        Load optimization configuration from YAML file.
        
        Args:
            config_path: Path to optimization_config.yaml. If None, uses default location.
            
        Returns:
            OptimizationConfig instance
        """
        if config_path is None:
            config_path = Path(__file__).parent / "optimization_config.yaml"
        
        if not config_path.exists():
            logger.warning(f"Optimization config file not found at {config_path}, using defaults")
            return cls()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            optimization_section = config_data.get('optimization', {})
            backtest_section = config_data.get('backtest', {})
            selection_section = config_data.get('selection', {})
            
            default_config = cls()
            
            # Parse optimization method
            opt_method_str = optimization_section.get('optimization_method', default_config.optimization_method.value)
            try:
                optimization_method = OptimizationMethod(opt_method_str)
            except ValueError:
                logger.warning(f"Invalid optimization_method: {opt_method_str}, using default")
                optimization_method = default_config.optimization_method
            
            # Parse optimization objective
            opt_obj_str = optimization_section.get('optimization_objective', default_config.optimization_objective.value)
            try:
                optimization_objective = OptimizationObjective(opt_obj_str)
            except ValueError:
                logger.warning(f"Invalid optimization_objective: {opt_obj_str}, using default")
                optimization_objective = default_config.optimization_objective
            
            # Parse wfa_start_date
            wfa_start_date_str = optimization_section.get('wfa_start_date')
            wfa_start_date = default_config.wfa_start_date
            if wfa_start_date_str:
                try:
                    if isinstance(wfa_start_date_str, str):
                        wfa_start_date = datetime.strptime(wfa_start_date_str, '%Y-%m-%d').date()
                    elif isinstance(wfa_start_date_str, date):
                        wfa_start_date = wfa_start_date_str
                except ValueError:
                    logger.warning(f"Invalid wfa_start_date: {wfa_start_date_str}, using default")
                    wfa_start_date = default_config.wfa_start_date
            
            return cls(
                max_iterations=optimization_section.get('max_iterations', default_config.max_iterations),
                convergence_threshold_pct=optimization_section.get('convergence_threshold_pct', default_config.convergence_threshold_pct),
                adjustment_step_pct=optimization_section.get('adjustment_step_pct', default_config.adjustment_step_pct),
                min_score_multiplier=optimization_section.get('min_score_multiplier', default_config.min_score_multiplier),
                max_score_multiplier=optimization_section.get('max_score_multiplier', default_config.max_score_multiplier),
                target_return_pct=optimization_section.get('target_return_pct', default_config.target_return_pct),
                target_win_rate_pct=optimization_section.get('target_win_rate_pct', default_config.target_win_rate_pct),
                max_drawdown_limit_pct=optimization_section.get('max_drawdown_limit_pct', default_config.max_drawdown_limit_pct),
                backtest_window_days=backtest_section.get('eval_window_days', default_config.backtest_window_days),
                rebalance_frequency_days=selection_section.get('rebalance_frequency_days', default_config.rebalance_frequency_days),
                enable_early_stopping=optimization_section.get('enable_early_stopping', default_config.enable_early_stopping),
                early_stopping_patience=optimization_section.get('early_stopping_patience', default_config.early_stopping_patience),
                random_seed=optimization_section.get('random_seed', default_config.random_seed),
                enable_sector_data=optimization_section.get('enable_sector_data', default_config.enable_sector_data),
                optimization_method=optimization_method,
                optimization_objective=optimization_objective,
                bayesian_n_calls=optimization_section.get('bayesian_n_calls', default_config.bayesian_n_calls),
                bayesian_n_initial_points=optimization_section.get('bayesian_n_initial_points', default_config.bayesian_n_initial_points),
                grid_search_step=optimization_section.get('grid_search_step', default_config.grid_search_step),
                normalize_weights=optimization_section.get('normalize_weights', default_config.normalize_weights),
                wfa_enabled=optimization_section.get('wfa_enabled', default_config.wfa_enabled),
                wfa_training_window_days=optimization_section.get('wfa_training_window_days', default_config.wfa_training_window_days),
                wfa_testing_window_days=optimization_section.get('wfa_testing_window_days', default_config.wfa_testing_window_days),
                wfa_step_days=optimization_section.get('wfa_step_days', default_config.wfa_step_days),
                wfa_num_windows=optimization_section.get('wfa_num_windows', default_config.wfa_num_windows),
                wfa_start_date=wfa_start_date,
                extra_config=config_data.get('extra_config', default_config.extra_config),
            )
        except Exception as e:
            logger.error(f"Failed to load optimization config from {config_path}: {e}, using defaults")
            return cls()


@dataclass
class WalkForwardWindow:
    """Single window in Walk-Forward Analysis."""
    
    window_id: str
    window_number: int
    training_start_date: date
    training_end_date: date
    testing_start_date: date
    testing_end_date: date
    training_result: Optional[IterationRecord] = None
    testing_result: Optional[BacktestResultSnapshot] = None
    bayesian_optimization_records: List[IterationRecord] = field(default_factory=list)
    best_params: Optional[List[float]] = None
    best_return: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['training_start_date'] = self.training_start_date.isoformat()
        data['training_end_date'] = self.training_end_date.isoformat()
        data['testing_start_date'] = self.testing_start_date.isoformat()
        data['testing_end_date'] = self.testing_end_date.isoformat()
        data['created_at'] = self.created_at.isoformat()
        if self.training_result:
            data['training_result'] = self.training_result.to_dict()
        if self.testing_result:
            data['testing_result'] = self.testing_result.to_dict()
        data['bayesian_optimization_records'] = [r.to_dict() for r in self.bayesian_optimization_records]
        if self.best_params:
            data['best_params'] = list(self.best_params)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WalkForwardWindow':
        """Create from dictionary."""
        data = data.copy()
        data['training_start_date'] = date.fromisoformat(data['training_start_date'])
        data['training_end_date'] = date.fromisoformat(data['training_end_date'])
        data['testing_start_date'] = date.fromisoformat(data['testing_start_date'])
        data['testing_end_date'] = date.fromisoformat(data['testing_end_date'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        training_result_data = data.pop('training_result', None)
        testing_result_data = data.pop('testing_result', None)
        bayesian_records_data = data.pop('bayesian_optimization_records', [])
        best_params = data.pop('best_params', None)
        best_return = data.pop('best_return', None)
        
        instance = cls(**data)
        if training_result_data:
            instance.training_result = IterationRecord.from_dict(training_result_data)
        if testing_result_data:
            instance.testing_result = BacktestResultSnapshot.from_dict(testing_result_data)
        instance.bayesian_optimization_records = [IterationRecord.from_dict(r) for r in bayesian_records_data]
        if best_params:
            instance.best_params = best_params
        if best_return:
            instance.best_return = best_return
        return instance


@dataclass
class WalkForwardResult:
    """Complete Walk-Forward Analysis result."""
    
    wfa_id: str
    windows: List[WalkForwardWindow]
    config: OptimizationConfig
    strategy_ids: List[str]
    aggregated_metrics: Dict[str, Any] = field(default_factory=dict)
    parameter_stability_metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['windows'] = [w.to_dict() for w in self.windows]
        data['config'] = self.config.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WalkForwardResult':
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['windows'] = [WalkForwardWindow.from_dict(w) for w in data['windows']]
        data['config'] = OptimizationConfig.from_dict(data['config'])
        return cls(**data)
    
    def compute_aggregated_metrics(self) -> None:
        """Compute aggregated metrics across all windows."""
        if not self.windows:
            return
        
        returns = []
        win_rates = []
        for window in self.windows:
            if window.testing_result:
                returns.append(window.testing_result.total_return_pct)
                if window.testing_result.win_rate_pct is not None:
                    win_rates.append(window.testing_result.win_rate_pct)
        
        if returns:
            self.aggregated_metrics['avg_return_pct'] = sum(returns) / len(returns)
            self.aggregated_metrics['return_std_pct'] = (
                sum((r - self.aggregated_metrics['avg_return_pct']) ** 2 for r in returns) / len(returns)
            ) ** 0.5
            self.aggregated_metrics['min_return_pct'] = min(returns)
            self.aggregated_metrics['max_return_pct'] = max(returns)
            self.aggregated_metrics['positive_window_count'] = sum(1 for r in returns if r > 0)
            self.aggregated_metrics['total_window_count'] = len(returns)
        
        if win_rates:
            self.aggregated_metrics['avg_win_rate_pct'] = sum(win_rates) / len(win_rates)
    
    def compute_parameter_stability(self) -> None:
        """Compute parameter stability metrics across all windows."""
        if not self.windows or len(self.windows) < 2:
            return
        
        windows_with_params = [w for w in self.windows if w.best_params is not None]
        if len(windows_with_params) < 2:
            return
        
        num_strategies = len(self.strategy_ids)
        param_history: List[List[float]] = [[] for _ in range(num_strategies)]
        
        for window in windows_with_params:
            if window.best_params:
                for i in range(min(num_strategies, len(window.best_params))):
                    param_history[i].append(window.best_params[i])
        
        stability_metrics = {}
        for i, strategy_id in enumerate(self.strategy_ids):
            params = param_history[i]
            if len(params) >= 2:
                mean_val = sum(params) / len(params)
                variance = sum((p - mean_val) ** 2 for p in params) / len(params)
                std_dev = variance ** 0.5
                coeff_var = (std_dev / abs(mean_val)) * 100 if mean_val != 0 else float('inf')
                
                stability_metrics[strategy_id] = {
                    'mean': mean_val,
                    'std_dev': std_dev,
                    'variance': variance,
                    'coeff_variation_pct': coeff_var,
                    'min': min(params),
                    'max': max(params),
                    'range': max(params) - min(params),
                    'num_windows': len(params)
                }
        
        self.parameter_stability_metrics = stability_metrics


class OptimizationEngine:
    """
    Optimization engine for historical data validation and suitable start date selection.
    Provides methods to validate historical data availability and find appropriate start dates for backtesting.
    """

    def __init__(
        self, 
        db_manager: Optional[DatabaseManager] = None,
        stock_selector_service: Optional[StockSelectorService] = None,
        nl_strategy_dir: Optional[Path] = None,
        python_strategy_dir: Optional[Path] = None,
    ):
        """
        Initialize the optimization engine with database access and stock selector service.

        Args:
            db_manager: Database manager instance (optional, uses singleton if not provided)
            stock_selector_service: StockSelectorService instance (optional, creates new if not provided)
            nl_strategy_dir: Directory for natural language YAML strategies
            python_strategy_dir: Directory for Python strategy modules
        """
        self.db_manager = db_manager or DatabaseManager.get_instance()
        self.stock_repo = StockRepository(self.db_manager)
        
        # Initialize stock selector service
        if stock_selector_service:
            self.stock_selector_service = stock_selector_service
        else:
            config = get_config()
            self.stock_selector_service = StockSelectorService(
                nl_strategy_dir=nl_strategy_dir,
                python_strategy_dir=python_strategy_dir,
                config=config,
            )
        
        logger.info("OptimizationEngine initialized")

    def validate_historical_date(
        self,
        target_date: date,
        eval_window_days: int,
        sample_stock_codes: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[date], str]:
        """
        Validate if a historical date has sufficient daily data available.

        This method checks if there are enough trading days before the target date
        to ensure historical context for strategy evaluation.

        Args:
            target_date: The date to validate
            eval_window_days: Number of days required for evaluation window
            sample_stock_codes: Optional list of stock codes to check data availability

        Returns:
            Tuple of (is_valid, earliest_available_date, message)
        """
        logger.info(f"Validating historical date: {target_date} with window: {eval_window_days} days")

        if sample_stock_codes is None:
            sample_stock_codes = self._get_sample_stock_codes()

        if not sample_stock_codes:
            logger.warning("No sample stock codes available for validation")
            return False, None, "No sample stock codes available"

        earliest_required_date = target_date - timedelta(days=eval_window_days + 30)

        has_data = False
        earliest_data_date = None

        for code in sample_stock_codes:
            try:
                data_range = self.stock_repo.get_range(
                    code,
                    earliest_required_date,
                    target_date
                )

                if len(data_range) >= eval_window_days:
                    has_data = True
                    if data_range:
                        current_earliest = min(d.date for d in data_range)
                        if earliest_data_date is None or current_earliest < earliest_data_date:
                            earliest_data_date = current_earliest
                    break
            except Exception as e:
                logger.warning(f"Failed to check data for stock {code}: {e}")
                continue

        if has_data:
            message = f"Historical data validation passed for {target_date}"
            logger.info(message)
            return True, earliest_data_date, message
        else:
            message = f"Insufficient historical data available for {target_date} with {eval_window_days} day window"
            logger.warning(message)
            return False, None, message

    def find_suitable_start_date(
        self,
        desired_start_date: date,
        end_date: date,
        eval_window_days: int,
        sample_stock_codes: Optional[List[str]] = None
    ) -> Tuple[Optional[date], str]:
        """
        Automatically find a suitable start date with sufficient historical data.

        This method searches backwards from the desired start date to find the earliest
        date that has sufficient historical data and forward data for backtesting.

        Args:
            desired_start_date: The initial desired start date
            end_date: The end date for the backtesting period
            eval_window_days: Number of days required for evaluation window
            sample_stock_codes: Optional list of stock codes to check data availability

        Returns:
            Tuple of (suitable_start_date, message)
        """
        logger.info(f"Finding suitable start date, desired: {desired_start_date}, end: {end_date}")

        if sample_stock_codes is None:
            sample_stock_codes = self._get_sample_stock_codes()

        current_date = desired_start_date
        max_search_days = 180

        for _ in range(max_search_days):
            if current_date >= end_date:
                message = "No suitable start date found before end date"
                logger.warning(message)
                return None, message

            is_valid, _, _ = self.validate_historical_date(
                current_date,
                eval_window_days,
                sample_stock_codes
            )

            if is_valid:
                has_forward, _, _ = self.check_forward_data_availability(
                    current_date,
                    eval_window_days,
                    sample_stock_codes
                )

                if has_forward:
                    message = f"Found suitable start date: {current_date}"
                    logger.info(message)
                    return current_date, message

            current_date = current_date - timedelta(days=1)

        message = f"No suitable start date found after searching {max_search_days} days"
        logger.warning(message)
        return None, message

    def check_forward_data_availability(
        self,
        start_date: date,
        eval_window_days: int,
        sample_stock_codes: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[date], str]:
        """
        Check if forward data is available for the backtesting window.

        This method verifies that there is sufficient data after the start date
        to complete the backtesting evaluation.

        Args:
            start_date: The start date of the backtest
            eval_window_days: Number of days required for evaluation window
            sample_stock_codes: Optional list of stock codes to check data availability

        Returns:
            Tuple of (is_available, latest_available_date, message)
        """
        logger.info(f"Checking forward data availability from {start_date} for {eval_window_days} days")

        if sample_stock_codes is None:
            sample_stock_codes = self._get_sample_stock_codes()

        if not sample_stock_codes:
            logger.warning("No sample stock codes available for forward data check")
            return False, None, "No sample stock codes available"

        latest_required_date = start_date + timedelta(days=eval_window_days + 30)

        has_data = False
        latest_data_date = None

        for code in sample_stock_codes:
            try:
                data_range = self.stock_repo.get_range(
                    code,
                    start_date,
                    latest_required_date
                )

                if len(data_range) >= eval_window_days:
                    has_data = True
                    if data_range:
                        current_latest = max(d.date for d in data_range)
                        if latest_data_date is None or current_latest > latest_data_date:
                            latest_data_date = current_latest
                    break
            except Exception as e:
                logger.warning(f"Failed to check forward data for stock {code}: {e}")
                continue

        if has_data:
            message = f"Forward data availability check passed for {start_date}"
            logger.info(message)
            return True, latest_data_date, message
        else:
            message = f"Insufficient forward data available from {start_date} for {eval_window_days} day window"
            logger.warning(message)
            return False, None, message

    def _get_sample_stock_codes(self, limit: int = 10) -> List[str]:
        """
        Get sample stock codes from the database for data availability checks.

        Args:
            limit: Maximum number of sample stock codes to return

        Returns:
            List of sample stock codes
        """
        try:
            with self.db_manager.get_session() as session:
                result = session.execute(
                    select(StockDaily.code)
                    .distinct()
                    .order_by(desc(StockDaily.date))
                    .limit(limit)
                ).scalars().all()
                return list(result)
        except Exception as e:
            logger.error(f"Failed to get sample stock codes: {e}")
            return []

    def get_latest_available_date(self) -> Tuple[Optional[date], str]:
        """
        Get the latest available date in the database.
        
        Returns:
            Tuple of (latest_date, message)
        """
        try:
            with self.db_manager.get_session() as session:
                stmt = (
                    select(StockDaily.date)
                    .distinct()
                    .order_by(desc(StockDaily.date))
                    .limit(1)
                )
                latest_date = session.execute(stmt).scalars().first()
            
            if latest_date:
                message = f"Latest available date: {latest_date}"
                logger.info(message)
                return latest_date, message
            else:
                message = "No data found in database"
                logger.warning(message)
                return None, message
        except Exception as e:
            message = f"Failed to get latest date: {e}"
            logger.error(message)
            return None, message

    def get_earliest_backtest_date(
        self,
        eval_window_days: int = 30,
        sample_stock_codes: Optional[List[str]] = None,
        start_date: Optional[date] = None
    ) -> Tuple[Optional[date], str]:
        """
        Find a suitable date for backtesting.
        
        If start_date is specified: use that date
        If start_date is not specified: use latest available date minus 180 days (stable date range)

        Args:
            eval_window_days: Number of days required for evaluation window
            sample_stock_codes: Optional list of stock codes to check
            start_date: Optional start date to begin searching from

        Returns:
            Tuple of (suitable_date, message)
        """
        logger.info(f"Finding suitable backtest date with {eval_window_days} day window")

        if sample_stock_codes is None:
            sample_stock_codes = self._get_sample_stock_codes(limit=20)

        if not sample_stock_codes:
            message = "No sample stock codes available"
            logger.warning(message)
            return None, message

        try:
            with self.db_manager.get_session() as session:
                stmt = (
                    select(StockDaily.date)
                    .distinct()
                    .order_by(StockDaily.date)
                )
                all_dates = session.execute(stmt).scalars().all()

            if not all_dates:
                message = "No historical data found in database"
                logger.warning(message)
                return None, message

            min_date = all_dates[0]
            max_date = all_dates[-1]

            logger.info(f"Database date range: {min_date} to {max_date}")

            # Determine the target date to start searching from
            if start_date:
                target_date = start_date
                logger.info(f"Using user-specified start date: {target_date}")
            else:
                # If no start date specified, use latest date minus 180 days for stable date range
                # This prevents the date from getting earlier every time we add more historical data
                target_date = max_date - timedelta(days=180)
                logger.info(f"No start date specified, using latest date minus 180 days: {target_date}")

            # Ensure target_date is within valid range
            earliest_possible = min_date + timedelta(days=eval_window_days + 60)
            latest_possible = max_date - timedelta(days=eval_window_days)
            
            if earliest_possible > latest_possible:
                message = f"Insufficient data - need at least {eval_window_days + 60 + eval_window_days} days of data"
                logger.warning(message)
                return None, message
            
            # Clamp target_date to valid range
            target_date = max(target_date, earliest_possible)
            target_date = min(target_date, latest_possible)
            
            logger.info(f"Searching for valid date around: {target_date}")

            # First try to find a valid date at or after target_date
            test_date = target_date
            max_search_days_forward = 90
            
            for i in range(max_search_days_forward):
                if test_date > latest_possible:
                    break

                is_valid, _, _ = self.validate_historical_date(
                    test_date,
                    eval_window_days,
                    sample_stock_codes
                )

                if is_valid:
                    has_forward, _, _ = self.check_forward_data_availability(
                        test_date,
                        eval_window_days,
                        sample_stock_codes
                    )

                    if has_forward:
                        message = f"Found suitable backtest date: {test_date}"
                        logger.info(message)
                        return test_date, message

                test_date += timedelta(days=1)
            
            # If no date found forward, try backward from target_date
            test_date = target_date - timedelta(days=1)
            max_search_days_backward = 90
            
            for i in range(max_search_days_backward):
                if test_date < earliest_possible:
                    break

                is_valid, _, _ = self.validate_historical_date(
                    test_date,
                    eval_window_days,
                    sample_stock_codes
                )

                if is_valid:
                    has_forward, _, _ = self.check_forward_data_availability(
                        test_date,
                        eval_window_days,
                        sample_stock_codes
                    )

                    if has_forward:
                        message = f"Found suitable backtest date: {test_date}"
                        logger.info(message)
                        return test_date, message

                test_date -= timedelta(days=1)

            message = f"No suitable date found after searching {max_search_days_forward + max_search_days_backward} days around {target_date}"
            logger.warning(message)
            return None, message

        except Exception as e:
            message = f"Failed to find suitable backtest date: {e}"
            logger.error(message)
            return None, message

    def _create_historical_data_provider(self, target_date: date, real_data_manager=None):
        """
        Create a historical data provider that provides data up to the target date.

        Args:
            target_date: The target date for historical data
            real_data_manager: Real data manager for fetching sector data when not in database

        Returns:
            A historical data provider instance
        """
        class HistoricalDataProvider:
            def __init__(self, db_manager: DatabaseManager, target_date: date, real_data_manager=None):
                self.db_manager = db_manager
                self.target_date = target_date
                self.real_data_manager = real_data_manager

            def _find_nearest_valid_date(self, stock_code: str, target: date) -> Optional[date]:
                """Find nearest valid date with data for the stock."""
                try:
                    with self.db_manager.get_session() as session:
                        # First try to find dates around target (30 days before and after)
                        from datetime import timedelta
                        search_start = target - timedelta(days=30)
                        search_end = target + timedelta(days=30)
                        
                        stmt = (
                            select(StockDaily.date)
                            .where(and_(
                                StockDaily.code == stock_code,
                                StockDaily.date >= search_start,
                                StockDaily.date <= search_end
                            ))
                            .order_by(StockDaily.date)
                        )
                        dates = session.execute(stmt).scalars().all()
                        
                        if dates:
                            # Find the closest date to target
                            closest_date = min(dates, key=lambda d: abs((d - target).days))
                            logger.debug(f"For {stock_code}, using nearest date {closest_date} instead of {target}")
                            return closest_date
                except Exception as e:
                    logger.debug(f"Failed to find nearest date for {stock_code}: {e}")
                return None

            def get_realtime_quote(self, stock_code: str):
                """
                Get historical quote data for a specific date, falling back to nearest date.

                Args:
                    stock_code: Stock code to get quote for

                Returns:
                    Mock quote object with historical data
                """
                try:
                    # First try target date
                    daily_data = self.db_manager.get_data_range(
                        stock_code, 
                        self.target_date, 
                        self.target_date
                    )
                    
                    if not daily_data:
                        # If target date not found, try nearest date
                        nearest_date = self._find_nearest_valid_date(stock_code, self.target_date)
                        if nearest_date:
                            daily_data = self.db_manager.get_data_range(
                                stock_code,
                                nearest_date,
                                nearest_date
                            )
                    
                    if daily_data:
                        data = daily_data[0]
                        class MockQuote:
                            def __init__(self, data):
                                self.price = data.close
                                self.change_pct = data.pct_chg
                                self.volume = data.volume
                                self.volume_ratio = data.volume_ratio
                                self.turnover_rate = None
                                self.name = stock_code
                            def has_basic_data(self):
                                return self.price is not None
                        return MockQuote(data)
                except Exception as e:
                    logger.warning(f"Failed to get historical quote for {stock_code} on {self.target_date}: {e}")
                return None

            def get_daily_data(self, stock_code: str, days: int = 60):
                """
                Get historical daily data up to the target date, falling back to nearest date.

                Args:
                    stock_code: Stock code to get data for
                    days: Number of days to retrieve

                Returns:
                    Tuple of (DataFrame, data_source)
                """
                try:
                    from datetime import timedelta
                    import pandas as pd
                    
                    # First, find if we have data up to target date
                    start_date = self.target_date - timedelta(days=days + 60)
                    
                    data_range = self.db_manager.get_data_range(
                        stock_code, 
                        start_date, 
                        self.target_date
                    )
                    
                    if not data_range:
                        # If no data up to target, try to find latest available date
                        with self.db_manager.get_session() as session:
                            stmt = (
                                select(func.max(StockDaily.date))
                                .where(and_(
                                    StockDaily.code == stock_code,
                                    StockDaily.date <= self.target_date
                                ))
                            )
                            latest_available = session.execute(stmt).scalar()
                            
                            if latest_available:
                                # Get data ending at latest available date
                                start_date_latest = latest_available - timedelta(days=days + 30)
                                data_range = self.db_manager.get_data_range(
                                    stock_code,
                                    start_date_latest,
                                    latest_available
                                )
                    
                    if data_range and len(data_range) > 0:
                        df = pd.DataFrame([item.to_dict() for item in data_range])
                        return df, "historical_database"
                except Exception as e:
                    logger.warning(f"Failed to get historical daily data for {stock_code}: {e}")
                return None, "historical_database"

            def get_stock_sectors(self, stock_code: str) -> Optional[List[str]]:
                """
                Get stock sectors - first check database, if not found, fetch from real data source.

                Args:
                    stock_code: Stock code to get sectors for

                Returns:
                    List of sector names, or None if not available
                """
                try:
                    logger.info(f"Checking database for {stock_code} sectors...")
                    sectors = self.db_manager.get_stock_sectors(stock_code)
                    
                    if sectors:
                        logger.info(f"Found {stock_code} sectors in database: {sectors}")
                        return sectors
                    
                    if self.real_data_manager:
                        logger.info(f"Sectors not in database, fetching {stock_code} sectors from real data source...")
                        sectors = self.real_data_manager.get_stock_sectors(stock_code)
                        
                        if sectors:
                            logger.info(f"Fetched {stock_code} sectors: {sectors}, saving to database...")
                            self.db_manager.save_stock_sectors(stock_code, sectors, "real_data_source")
                            return sectors
                    
                    logger.warning(f"Could not get {stock_code} sectors from database or real data source")
                except Exception as e:
                    logger.error(f"Failed to get {stock_code} sectors: {e}")
                
                return None

            def get_sector_rankings(self, n: int = 5) -> Tuple[List[Dict], List[Dict]]:
                """
                Get sector rankings - first check database, if not found, fetch from real data source and save.

                Args:
                    n: Number of top/bottom sectors to return

                Returns:
                    Tuple of (top sectors list, bottom sectors list)
                """
                try:
                    from sqlalchemy import select, and_, desc

                    with self.db_manager.get_session() as session:
                        stmt = (
                            select(SectorDaily)
                            .where(SectorDaily.date == self.target_date)
                            .order_by(desc(SectorDaily.change_pct))
                        )
                        all_sectors = session.execute(stmt).scalars().all()

                        if all_sectors:
                            logger.info(f"Found {len(all_sectors)} sector records in database for {self.target_date}")
                            top_sectors = [
                                {'name': s.name, 'change_pct': s.change_pct}
                                for s in all_sectors[:n]
                                if s.change_pct is not None
                            ]
                            bottom_sectors = [
                                {'name': s.name, 'change_pct': s.change_pct}
                                for s in reversed(all_sectors[-n:])
                                if s.change_pct is not None
                            ]
                            return top_sectors, bottom_sectors
                except Exception as e:
                    logger.warning(f"Failed to get sector rankings from database: {e}")

                if self.real_data_manager:
                    try:
                        logger.info(f"Sector rankings not in database, fetching from real data source...")
                        top_sectors, bottom_sectors = self.real_data_manager.get_sector_rankings(n)

                        if top_sectors or bottom_sectors:
                            logger.info(f"Saving {len(top_sectors) + len(bottom_sectors)} sector rankings to database for {self.target_date}...")
                            self.db_manager.save_sector_rankings(
                                top_sectors,
                                bottom_sectors,
                                date=self.target_date,
                                data_source="real_data_source"
                            )

                        return top_sectors, bottom_sectors
                    except Exception as e:
                        logger.error(f"Failed to get sector rankings from real data source: {e}")

                return [], []

            def get_sector_change_pct(self, sector_name: str) -> Optional[float]:
                """
                Get sector change percentage for the target date from database.

                Args:
                    sector_name: Name of the sector

                Returns:
                    Change percentage, or None if not available
                """
                try:
                    sector_data = self.db_manager.get_sector_daily(sector_name, self.target_date)
                    if sector_data:
                        logger.debug(f"Found sector {sector_name} change_pct={sector_data.change_pct} for {self.target_date}")
                        return sector_data.change_pct
                except Exception as e:
                    logger.warning(f"Failed to get sector change_pct for {sector_name}: {e}")

                return None

        return HistoricalDataProvider(self.db_manager, target_date, real_data_manager)

    def _convert_candidate_to_record(self, candidate: StockCandidate) -> StockSelectionRecord:
        """
        Convert a StockCandidate to a StockSelectionRecord.

        Args:
            candidate: StockCandidate to convert

        Returns:
            Converted StockSelectionRecord
        """
        strategy_matches = []
        for match in candidate.strategy_matches:
            strategy_matches.append({
                "strategy_id": match.strategy_id,
                "strategy_name": match.strategy_name,
                "matched": match.matched,
                "score": match.score,
                "raw_score": match.raw_score,
                "reason": match.reason,
                "match_details": match.match_details
            })

        return StockSelectionRecord(
            code=candidate.code,
            name=candidate.name,
            buy_price=candidate.current_price,
            match_score=candidate.match_score,
            strategy_matches=strategy_matches,
            extra_data=candidate.extra_data
        )

    def _screen_stock_historical(
        self,
        stock_code: str,
        target_date: date,
        strategy_ids: Optional[List[str]] = None,
    ) -> Optional[StockCandidate]:
        """
        Screen a single stock using historical data.

        Args:
            stock_code: Stock code to screen
            target_date: Target date for historical screening
            strategy_ids: Optional list of strategy IDs to use

        Returns:
            StockCandidate if matched, None otherwise
        """
        try:
            strategy_manager = self.stock_selector_service.strategy_manager
            
            # Get strategies to use
            strategies_to_use = []
            if strategy_ids:
                for sid in strategy_ids:
                    strategy = strategy_manager._strategies.get(sid)
                    if strategy:
                        strategies_to_use.append(strategy)
                    else:
                        logger.warning("Unknown strategy ID: %s", sid)
            else:
                strategies_to_use = strategy_manager.get_active_strategies()

            if not strategies_to_use:
                return None

            # Create historical data provider with real data manager for sector fetching
            real_data_manager = strategy_manager._data_provider
            historical_data_provider = self._create_historical_data_provider(target_date, real_data_manager)
            
            # Store original providers for restoration
            original_strategy_providers = {}
            original_sector_data_manager = None
            
            # Temporarily replace data providers for all strategies
            for strategy in strategies_to_use:
                original_strategy_providers[strategy.id] = strategy._data_provider
                strategy._data_provider = historical_data_provider
            
            # Also replace sector manager's data manager if it exists
            if strategy_manager._sector_manager:
                original_sector_data_manager = strategy_manager._sector_manager._data_manager
                strategy_manager._sector_manager._data_manager = historical_data_provider
            
            # Execute each strategy directly (bypassing execute_strategies' caching logic)
            matches = []
            for strategy in strategies_to_use:
                try:
                    match = strategy.select(stock_code, None)
                    matches.append(match)
                except Exception as e:
                    logger.error("Strategy %s failed for %s: %s", strategy.id, stock_code, e)
                    matches.append(
                        StrategyMatch(
                            strategy_id=strategy.id,
                            strategy_name=strategy.display_name,
                            matched=False,
                            reason=f"Strategy execution error: {str(e)}",
                        )
                    )
            
            if not matches:
                # Restore original providers before returning
                for strategy in strategies_to_use:
                    if strategy.id in original_strategy_providers:
                        strategy._data_provider = original_strategy_providers[strategy.id]
                if original_sector_data_manager and strategy_manager._sector_manager:
                    strategy_manager._sector_manager._data_manager = original_sector_data_manager
                return None

            # Get stock name and price from historical data
            stock_name = stock_code
            current_price = 0.0
            try:
                daily_data = self.db_manager.get_data_range(
                    stock_code, 
                    target_date, 
                    target_date
                )
                if daily_data:
                    data = daily_data[0]
                    current_price = data.close
            except Exception as e:
                logger.warning(f"Failed to get historical price for {stock_code}: {e}")

            # Create candidate
            candidate = StockCandidate(
                code=stock_code,
                name=stock_name,
                current_price=current_price,
            )

            for match in matches:
                candidate.add_strategy_match(match)

            # Restore original providers
            for strategy in strategies_to_use:
                if strategy.id in original_strategy_providers:
                    strategy._data_provider = original_strategy_providers[strategy.id]
            if original_sector_data_manager and strategy_manager._sector_manager:
                strategy_manager._sector_manager._data_manager = original_sector_data_manager

            return candidate

        except Exception as e:
            # Ensure original providers are restored even if there's an error
            try:
                strategy_manager = self.stock_selector_service.strategy_manager
                if 'original_strategy_providers' in locals():
                    strategies_to_use = locals().get('strategies_to_use', [])
                    for strategy in strategies_to_use:
                        if strategy.id in original_strategy_providers:
                            strategy._data_provider = original_strategy_providers[strategy.id]
                if 'original_sector_data_manager' in locals() and original_sector_data_manager and strategy_manager._sector_manager:
                    strategy_manager._sector_manager._data_manager = original_sector_data_manager
            except:
                pass
            logger.error(f"Failed to screen stock {stock_code} historically: {e}")
            return None

    def run_historical_selection(
        self,
        target_date: date,
        strategy_ids: List[str],
        top_n: int,
        stock_codes: Optional[List[str]] = None,
    ) -> StockSelectionResult:
        """
        Run stock selection strategies on a specified historical date.

        Args:
            target_date: Target date for historical selection
            strategy_ids: List of strategy IDs to use
            top_n: Number of top stocks to return
            stock_codes: Optional list of stock codes to screen (uses all stocks if not provided)

        Returns:
            StockSelectionResult with the selected stocks

        Raises:
            ValueError: If validation fails
            Exception: If an unexpected error occurs
        """
        logger.info(f"Starting historical selection for date {target_date}, strategies: {strategy_ids}, top_n: {top_n}")

        # Validate target date
        if target_date > date.today():
            raise ValueError(f"Target date {target_date} cannot be in the future")

        # Validate historical data availability
        is_valid, _, msg = self.validate_historical_date(target_date, 30)
        if not is_valid:
            logger.warning(f"Historical date validation warning: {msg}")

        # Get stock codes to screen
        if stock_codes is None:
            stock_codes = get_all_stock_codes()
            logger.info(f"Using complete stock pool with {len(stock_codes)} stocks")
        else:
            logger.info(f"Using provided stock list with {len(stock_codes)} stocks")

        candidates: List[StockCandidate] = []
        total_candidates = len(stock_codes)
        min_match_score = float('inf')

        # Process stocks in parallel
        max_workers = min(10, len(stock_codes))
        logger.info(f"Using thread pool with {max_workers} workers for historical screening")

        import threading
        progress_lock = threading.Lock()
        processed_count = 0
        progress = ProgressBar(total=total_candidates, description="Screening Stocks")

        def process_stock(code):
            nonlocal processed_count
            try:
                result = self._screen_stock_historical(code, target_date, strategy_ids)
                
                # Update progress
                with progress_lock:
                    processed_count += 1
                    progress.update(processed_count)
                
                return result
            except Exception as e:
                logger.error(f"Failed to screen stock {code} historically: {e}")
                with progress_lock:
                    processed_count += 1
                    progress.update(processed_count)
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_stock, stock_codes))

        progress.finish(f"Screened {processed_count} stocks")

        # Collect valid candidates
        for result in results:
            if result:
                candidates.append(result)
                if result.match_score < min_match_score:
                    min_match_score = result.match_score

        # If no valid candidates found, set min_match_score to 0
        if min_match_score == float('inf'):
            min_match_score = 0.0

        # Rank candidates
        logger.info(f"Ranking {len(candidates)} valid candidates")
        strategy_manager = self.stock_selector_service.strategy_manager
        ranked_candidates = strategy_manager.rank_candidates(candidates, top_n=top_n)

        # Convert to StockSelectionRecord format
        selected_stocks = [self._convert_candidate_to_record(candidate) for candidate in ranked_candidates]

        # Create result
        result = StockSelectionResult(
            selection_date=datetime.combine(target_date, datetime.min.time()),
            selected_stocks=selected_stocks,
            total_candidates=total_candidates,
            min_match_score=min_match_score,
            top_n=top_n,
            extra_data={
                "strategy_ids": strategy_ids,
                "target_date": target_date.isoformat()
            }
        )

        logger.info(f"Historical selection completed: {len(selected_stocks)} stocks selected out of {total_candidates} candidates")
        return result

    def run_historical_backtest(
        self,
        selection_result: StockSelectionResult,
        start_date: date,
        eval_window_days: int,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        neutral_band_pct: float = 2.0,
        engine_version: str = "v1",
    ) -> BacktestResultSnapshot:
        """
        Run historical backtest on stock selection result.
        
        Args:
            selection_result: StockSelectionResult containing selected stocks
            start_date: Start date for backtesting
            eval_window_days: Evaluation window in days
            stop_loss_pct: Stop loss percentage (optional)
            take_profit_pct: Take profit percentage (optional)
            neutral_band_pct: Neutral band percentage for outcome classification
            engine_version: Backtest engine version
            
        Returns:
            BacktestResultSnapshot with comprehensive backtest metrics
            
        Raises:
            ValueError: If input parameters are invalid
            Exception: If backtest execution fails
        """
        backtest_id = str(uuid.uuid4())
        logger.info(f"Starting historical backtest: {backtest_id}, start_date: {start_date}, eval_window: {eval_window_days} days")
        
        try:
            if not selection_result.selected_stocks:
                logger.warning("No selected stocks in selection result for backtesting")
                return self._create_empty_backtest_snapshot(
                    backtest_id=backtest_id,
                    start_date=start_date,
                    eval_window_days=eval_window_days,
                    engine_version=engine_version
                )
            
            end_date = start_date + timedelta(days=eval_window_days + 30)
            
            eval_config = EvaluationConfig(
                eval_window_days=eval_window_days,
                neutral_band_pct=neutral_band_pct,
                engine_version=engine_version
            )
            
            backtest_results = []
            stock_returns = []
            simulated_returns = []
            
            total_stocks = len(selection_result.selected_stocks)
            progress = ProgressBar(total=total_stocks, description="Backtesting Stocks")
            
            for stock_record in selection_result.selected_stocks:
                try:
                    backtest_result = self._backtest_single_stock(
                        stock_record=stock_record,
                        start_date=start_date,
                        eval_config=eval_config,
                        stop_loss_pct=stop_loss_pct,
                        take_profit_pct=take_profit_pct
                    )
                    
                    if backtest_result:
                        backtest_results.append(backtest_result)
                        if backtest_result.stock_return_pct is not None:
                            stock_returns.append(backtest_result.stock_return_pct)
                        if backtest_result.simulated_return_pct is not None:
                            simulated_returns.append(backtest_result.simulated_return_pct)
                        
                except Exception as e:
                    logger.warning(f"Failed to backtest stock {stock_record.code}: {e}")
                    continue
                
                progress.increment()
            
            progress.finish(f"Backtested {len(backtest_results)} stocks successfully")
            
            if not backtest_results:
                logger.warning("No valid backtest results generated")
                return self._create_empty_backtest_snapshot(
                    backtest_id=backtest_id,
                    start_date=start_date,
                    eval_window_days=eval_window_days,
                    engine_version=engine_version
                )
            
            snapshot = self._aggregate_backtest_results(
                backtest_id=backtest_id,
                backtest_results=backtest_results,
                stock_returns=stock_returns,
                simulated_returns=simulated_returns,
                start_date=start_date,
                eval_window_days=eval_window_days,
                engine_version=engine_version,
                selection_result=selection_result
            )
            
            logger.info(f"Historical backtest completed: {backtest_id}, total_return: {snapshot.total_return_pct:.2f}%, win_rate: {snapshot.win_rate_pct:.2f}%")
            return snapshot
            
        except Exception as e:
            logger.error(f"Historical backtest failed: {backtest_id}, error: {e}")
            raise

    def _backtest_single_stock(
        self,
        stock_record: StockSelectionRecord,
        start_date: date,
        eval_config: EvaluationConfig,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
    ) -> Optional[BacktestResult]:
        """
        Backtest a single stock.
        
        Args:
            stock_record: StockSelectionRecord for the stock
            start_date: Start date for backtesting
            eval_config: Evaluation configuration
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            
        Returns:
            BacktestResult or None if failed
        """
        logger.debug(f"Backtesting {stock_record.code} from {start_date}, eval_window={eval_config.eval_window_days}")
        
        start_daily = self.stock_repo.get_start_daily(
            code=stock_record.code,
            analysis_date=start_date
        )
        
        if start_daily is None or start_daily.close is None:
            logger.debug(f"No start daily data for {stock_record.code} on {start_date}, trying to find nearby valid date...")
            try:
                with self.db_manager.get_session() as session:
                    # First, try to find dates before or on start_date (prefer before)
                    search_start = start_date - timedelta(days=90)
                    
                    stmt = (
                        select(StockDaily)
                        .where(StockDaily.code == stock_record.code)
                        .where(StockDaily.date >= search_start)
                        .where(StockDaily.date <= start_date)
                        .where(StockDaily.close.isnot(None))
                        .order_by(StockDaily.date.desc())
                    )
                    
                    results = session.execute(stmt).scalars().all()
                    logger.debug(f"Found {len(results)} candidate dates before/on {start_date} for {stock_record.code}")
                    
                    if results:
                        start_daily = results[0]
                        logger.debug(f"Found valid data for {stock_record.code} on {start_daily.date}")
                    else:
                        # If no dates before, try dates after (as fallback)
                        logger.debug(f"No dates found before/on {start_date}, trying dates after...")
                        search_end = start_date + timedelta(days=30)
                        
                        stmt_after = (
                            select(StockDaily)
                            .where(StockDaily.code == stock_record.code)
                            .where(StockDaily.date >= start_date)
                            .where(StockDaily.date <= search_end)
                            .where(StockDaily.close.isnot(None))
                            .order_by(StockDaily.date.asc())
                        )
                        
                        results_after = session.execute(stmt_after).scalars().all()
                        if results_after:
                            start_daily = results_after[0]
                            logger.debug(f"Found valid data after {start_date} for {stock_record.code} on {start_daily.date}")
            except Exception as e:
                logger.debug(f"Failed to find nearby date for {stock_record.code}: {e}", exc_info=True)
        
        if start_daily is None or start_daily.close is None:
            logger.warning(f"No start daily data for {stock_record.code} on or before {start_date}")
            return None
        
        logger.debug(f"Got start daily for {stock_record.code}: date={start_daily.date}, close={start_daily.close}")
        
        forward_bars = self.stock_repo.get_forward_bars(
            code=stock_record.code,
            analysis_date=start_daily.date,
            eval_window_days=eval_config.eval_window_days
        )
        
        logger.debug(f"Got {len(forward_bars)} forward bars for {stock_record.code}, need {eval_config.eval_window_days}")
        
        if len(forward_bars) < eval_config.eval_window_days:
            logger.warning(f"Insufficient forward bars for {stock_record.code}: got {len(forward_bars)}, need {eval_config.eval_window_days}")
            return None
        
        stop_loss = None
        take_profit = None
        if stop_loss_pct is not None:
            stop_loss = start_daily.close * (1 - stop_loss_pct / 100)
        if take_profit_pct is not None:
            take_profit = start_daily.close * (1 + take_profit_pct / 100)
        
        operation_advice = "买入"
        
        evaluation = BacktestEngine.evaluate_single(
            operation_advice=operation_advice,
            analysis_date=start_daily.date,
            start_price=float(start_daily.close),
            forward_bars=forward_bars,
            stop_loss=stop_loss,
            take_profit=take_profit,
            config=eval_config
        )
        
        backtest_result = BacktestResult(
            code=stock_record.code,
            analysis_date=evaluation.get("analysis_date"),
            eval_window_days=evaluation.get("eval_window_days"),
            engine_version=evaluation.get("engine_version"),
            eval_status=evaluation.get("eval_status"),
            evaluated_at=datetime.now(),
            operation_advice=evaluation.get("operation_advice"),
            position_recommendation=evaluation.get("position_recommendation"),
            start_price=evaluation.get("start_price"),
            end_close=evaluation.get("end_close"),
            max_high=evaluation.get("max_high"),
            min_low=evaluation.get("min_low"),
            stock_return_pct=evaluation.get("stock_return_pct"),
            direction_expected=evaluation.get("direction_expected"),
            direction_correct=evaluation.get("direction_correct"),
            outcome=evaluation.get("outcome"),
            stop_loss=evaluation.get("stop_loss"),
            take_profit=evaluation.get("take_profit"),
            hit_stop_loss=evaluation.get("hit_stop_loss"),
            hit_take_profit=evaluation.get("hit_take_profit"),
            first_hit=evaluation.get("first_hit"),
            first_hit_date=evaluation.get("first_hit_date"),
            first_hit_trading_days=evaluation.get("first_hit_trading_days"),
            simulated_entry_price=evaluation.get("simulated_entry_price"),
            simulated_exit_price=evaluation.get("simulated_exit_price"),
            simulated_exit_reason=evaluation.get("simulated_exit_reason"),
            simulated_return_pct=evaluation.get("simulated_return_pct"),
        )
        
        return backtest_result

    def _aggregate_backtest_results(
        self,
        backtest_id: str,
        backtest_results: List[BacktestResult],
        stock_returns: List[float],
        simulated_returns: List[float],
        start_date: date,
        eval_window_days: int,
        engine_version: str,
        selection_result: Optional[StockSelectionResult] = None,
    ) -> BacktestResultSnapshot:
        """
        Aggregate individual backtest results into BacktestResultSnapshot.
        
        Args:
            backtest_id: Unique backtest ID
            backtest_results: List of individual BacktestResult objects
            stock_returns: List of stock return percentages
            simulated_returns: List of simulated return percentages
            start_date: Start date of backtest
            eval_window_days: Evaluation window in days
            engine_version: Engine version
            selection_result: StockSelectionResult with strategy matches
            
        Returns:
            BacktestResultSnapshot with aggregated metrics
        """
        completed_results = [r for r in backtest_results if r.eval_status == "completed"]
        
        total_trades = len(completed_results)
        winning_trades = sum(1 for r in completed_results if r.outcome == "win")
        losing_trades = sum(1 for r in completed_results if r.outcome == "loss")
        neutral_trades = sum(1 for r in completed_results if r.outcome == "neutral")
        
        win_rate_pct = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        total_return_pct = sum(simulated_returns) if simulated_returns else 0.0
        
        max_drawdown_pct = self._calculate_max_drawdown(stock_returns)
        
        direction_correct_count = sum(1 for r in completed_results if r.direction_correct is True)
        direction_total_count = sum(1 for r in completed_results if r.direction_correct is not None)
        direction_accuracy_pct = (direction_correct_count / direction_total_count * 100) if direction_total_count > 0 else None
        
        avg_stock_return_pct = sum(stock_returns) / len(stock_returns) if stock_returns else None
        avg_simulated_return_pct = sum(simulated_returns) / len(simulated_returns) if simulated_returns else None
        
        stop_loss_applicable = [r for r in completed_results if r.hit_stop_loss is not None]
        stop_loss_trigger_rate_pct = (sum(1 for r in stop_loss_applicable if r.hit_stop_loss) / len(stop_loss_applicable) * 100) if stop_loss_applicable else None
        
        take_profit_applicable = [r for r in completed_results if r.hit_take_profit is not None]
        take_profit_trigger_rate_pct = (sum(1 for r in take_profit_applicable if r.hit_take_profit) / len(take_profit_applicable) * 100) if take_profit_applicable else None
        
        days_to_first_hit_list = [r.first_hit_trading_days for r in completed_results if r.first_hit_trading_days is not None]
        avg_days_to_first_hit = sum(days_to_first_hit_list) / len(days_to_first_hit_list) if days_to_first_hit_list else None
        
        end_date = start_date + timedelta(days=eval_window_days)
        
        stock_details = []
        if selection_result:
            stock_record_map = {sr.code: sr for sr in selection_result.selected_stocks}
            for result in backtest_results:
                if result.code in stock_record_map:
                    stock_record = stock_record_map[result.code]
                    detail = StockBacktestDetail(
                        code=result.code,
                        name=stock_record.name,
                        stock_return_pct=result.stock_return_pct or 0.0,
                        simulated_return_pct=result.simulated_return_pct,
                        outcome=result.outcome,
                        strategy_matches=stock_record.strategy_matches
                    )
                    stock_details.append(detail)
        
        return BacktestResultSnapshot(
            backtest_id=backtest_id,
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.min.time()),
            total_return_pct=total_return_pct,
            win_rate_pct=win_rate_pct,
            max_drawdown_pct=max_drawdown_pct,
            direction_accuracy_pct=direction_accuracy_pct,
            avg_stock_return_pct=avg_stock_return_pct,
            avg_simulated_return_pct=avg_simulated_return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            neutral_trades=neutral_trades,
            stop_loss_trigger_rate_pct=stop_loss_trigger_rate_pct,
            take_profit_trigger_rate_pct=take_profit_trigger_rate_pct,
            avg_days_to_first_hit=avg_days_to_first_hit,
            eval_window_days=eval_window_days,
            engine_version=engine_version,
            extra_metrics={},
            created_at=datetime.now(),
            stock_details=stock_details
        )

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """
        Calculate maximum drawdown from return series.
        
        Args:
            returns: List of return percentages
            
        Returns:
            Maximum drawdown percentage
        """
        if not returns:
            return 0.0
        
        cumulative = [0.0]
        for r in returns:
            cumulative.append(cumulative[-1] + r)
        
        max_drawdown = 0.0
        peak = cumulative[0]
        
        for value in cumulative:
            if value > peak:
                peak = value
            drawdown = peak - value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown

    def _create_empty_backtest_snapshot(
        self,
        backtest_id: str,
        start_date: date,
        eval_window_days: int,
        engine_version: str,
    ) -> BacktestResultSnapshot:
        """
        Create an empty backtest result snapshot.
        
        Args:
            backtest_id: Unique backtest ID
            start_date: Start date
            eval_window_days: Evaluation window days
            engine_version: Engine version
            
        Returns:
            Empty BacktestResultSnapshot
        """
        end_date = start_date + timedelta(days=eval_window_days)
        return BacktestResultSnapshot(
            backtest_id=backtest_id,
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.min.time()),
            total_return_pct=0.0,
            win_rate_pct=0.0,
            max_drawdown_pct=0.0,
            direction_accuracy_pct=None,
            avg_stock_return_pct=None,
            avg_simulated_return_pct=None,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            neutral_trades=0,
            stop_loss_trigger_rate_pct=None,
            take_profit_trigger_rate_pct=None,
            avg_days_to_first_hit=None,
            eval_window_days=eval_window_days,
            engine_version=engine_version,
            extra_metrics={},
            created_at=datetime.now()
        )

    def optimize_strategy_params(
        self,
        backtest_result: BacktestResultSnapshot,
        current_params: StrategyParamsSnapshot,
        config: OptimizationConfig,
    ) -> Tuple[StrategyParamsSnapshot, OptimizationDirection, str]:
        """
        Optimize strategy parameters based on backtest results and strategy contribution.
        
        Args:
            backtest_result: Backtest result snapshot with stock details
            current_params: Current strategy parameters
            config: Optimization configuration
            
        Returns:
            Tuple of (optimized_params, optimization_direction, optimization_reason)
        """
        strategy_id = current_params.strategy_id
        logger.info(f"Optimizing strategy params for {strategy_id}")
        
        optimized_params = StrategyParamsSnapshot(
            strategy_id=strategy_id,
            strategy_name=current_params.strategy_name,
            score_multiplier=current_params.score_multiplier,
            max_raw_score=current_params.max_raw_score,
            enabled=current_params.enabled,
            custom_params=current_params.custom_params.copy(),
            created_at=datetime.now()
        )
        
        direction = OptimizationDirection.NO_CHANGE
        reason = "No optimization needed - performance is stable"
        
        adjustment_step = config.adjustment_step_pct / 100.0
        
        try:
            # Calculate strategy-specific performance based on stock details
            strategy_contributed_stocks = []
            strategy_weighted_return = 0.0
            strategy_total_weight = 0.0
            strategy_win_count = 0
            strategy_loss_count = 0
            
            for stock_detail in backtest_result.stock_details:
                # Check if this strategy contributed to this stock
                for match in stock_detail.strategy_matches:
                    if match.get('strategy_id') == strategy_id and match.get('matched', False):
                        # Use the strategy's score as weight
                        score = match.get('score', 0.0)
                        if score > 0:
                            strategy_contributed_stocks.append(stock_detail)
                            strategy_weighted_return += stock_detail.stock_return_pct * score
                            strategy_total_weight += score
                            
                            if stock_detail.outcome == 'win':
                                strategy_win_count += 1
                            elif stock_detail.outcome == 'loss':
                                strategy_loss_count += 1
                        break
            
            # If strategy didn't contribute to any stocks, use overall performance
            if not strategy_contributed_stocks:
                logger.info(f"Strategy {strategy_id} did not contribute to any selected stocks, using overall performance")
                avg_return = backtest_result.total_return_pct
                win_rate = backtest_result.win_rate_pct
            else:
                # Calculate strategy-specific metrics
                avg_return = strategy_weighted_return / strategy_total_weight if strategy_total_weight > 0 else 0.0
                total_trades = strategy_win_count + strategy_loss_count
                win_rate = (strategy_win_count / total_trades * 100) if total_trades > 0 else 0.0
                
                logger.info(f"Strategy {strategy_id} contributed to {len(strategy_contributed_stocks)} stocks: "
                           f"avg_return={avg_return:.2f}%, win_rate={win_rate:.2f}%")
            
            # Make adjustment decision based on strategy-specific performance
            if avg_return > 0:
                if win_rate >= 50.0:
                    # Calculate dynamic adjustment step based on performance - more aggressive
                    performance_score = (avg_return / 5.0) * (win_rate / 50.0)
                    performance_score = max(0.5, min(3.0, performance_score))
                    dynamic_adjustment = adjustment_step * performance_score
                    
                    new_multiplier = min(
                        config.max_score_multiplier,
                        optimized_params.score_multiplier * (1 + dynamic_adjustment)
                    )
                    if new_multiplier != optimized_params.score_multiplier:
                        optimized_params.score_multiplier = new_multiplier
                        direction = OptimizationDirection.INCREASE_SCORE_MULTIPLIER
                        reason = f"Good strategy performance (return: {avg_return:.2f}%, win_rate: {win_rate:.2f}%, adj_step: {dynamic_adjustment*100:.1f}%) - increasing score multiplier"
                        logger.info(reason)
            else:
                if win_rate < 50.0 or backtest_result.max_drawdown_pct > 20.0:
                    # Calculate dynamic adjustment step based on how bad the performance is - more aggressive
                    loss_magnitude = abs(avg_return) / 5.0
                    win_rate_penalty = (50.0 - win_rate) / 25.0 if win_rate < 50.0 else 0
                    performance_penalty = loss_magnitude + win_rate_penalty
                    performance_penalty = max(0.5, min(3.0, performance_penalty))
                    dynamic_adjustment = adjustment_step * performance_penalty
                    
                    new_multiplier = max(
                        config.min_score_multiplier,
                        optimized_params.score_multiplier * (1 - dynamic_adjustment)
                    )
                    if new_multiplier != optimized_params.score_multiplier:
                        optimized_params.score_multiplier = new_multiplier
                        direction = OptimizationDirection.DECREASE_SCORE_MULTIPLIER
                        reason = f"Poor strategy performance (return: {avg_return:.2f}%, win_rate: {win_rate:.2f}%, drawdown: {backtest_result.max_drawdown_pct:.2f}%, adj_step: {dynamic_adjustment*100:.1f}%) - decreasing score multiplier"
                        logger.info(reason)
                        
        except Exception as e:
            logger.error(f"Error during parameter optimization: {e}")
            reason = f"Optimization failed with error: {str(e)}"
            
        return optimized_params, direction, reason

    def check_convergence(
        self,
        iteration_records: List[IterationRecord],
        config: OptimizationConfig,
    ) -> Tuple[bool, str]:
        """
        Check if optimization has converged.
        
        Args:
            iteration_records: List of iteration records
            config: Optimization configuration
            
        Returns:
            Tuple of (has_converged, message)
        """
        if len(iteration_records) >= config.max_iterations:
            return True, f"Reached maximum iterations ({config.max_iterations})"
            
        if len(iteration_records) < 3:
            return False, "Not enough iterations to check convergence"
            
        recent_returns = [
            r.backtest_result.total_return_pct 
            for r in iteration_records[-3:]
        ]
        
        return_std = sum(abs(r - recent_returns[-1]) for r in recent_returns) / len(recent_returns)
        
        if return_std < config.convergence_threshold_pct:
            return True, f"Convergence reached - return std: {return_std:.4f}% < threshold: {config.convergence_threshold_pct}%"
            
        return False, f"Not converged - return std: {return_std:.4f}%"

    def run_optimization_iteration(
        self,
        start_date: date,
        strategy_ids: List[str],
        config: OptimizationConfig,
        current_params: Optional[Dict[str, StrategyParamsSnapshot]] = None,
        iteration_number: int = 1,
        parent_iteration_id: Optional[str] = None,
        top_n: int = 5,
        stock_codes: Optional[List[str]] = None,
    ) -> Optional[IterationRecord]:
        """
        Run a single optimization iteration.
        
        Args:
            start_date: Start date for the iteration
            strategy_ids: List of strategy IDs to optimize
            config: Optimization configuration
            current_params: Current strategy parameters (optional)
            iteration_number: Current iteration number
            parent_iteration_id: Parent iteration ID (optional)
            top_n: Number of top stocks to select
            
        Returns:
            IterationRecord or None if failed
        """
        iteration_id = str(uuid.uuid4())
        logger.info(f"Starting optimization iteration {iteration_number} (ID: {iteration_id})")
        
        try:
            if current_params is None:
                current_params = {}
                for sid in strategy_ids:
                    strategy = self.stock_selector_service.strategy_manager.get_strategy(sid)
                    if strategy:
                        current_params[sid] = StrategyParamsSnapshot(
                            strategy_id=sid,
                            strategy_name=strategy.display_name,
                            score_multiplier=strategy.score_multiplier,
                            max_raw_score=100.0,
                            enabled=True,
                            custom_params={}
                        )
            
            for sid, params in current_params.items():
                self.stock_selector_service.strategy_manager.set_strategy_multiplier(
                    sid, params.score_multiplier
                )
            
            selection_result = self.run_historical_selection(
                target_date=start_date,
                strategy_ids=strategy_ids,
                top_n=top_n,
                stock_codes=stock_codes
            )
            
            backtest_result = self.run_historical_backtest(
                selection_result=selection_result,
                start_date=start_date,
                eval_window_days=config.backtest_window_days
            )
            
            optimized_params_dict = {}
            direction = OptimizationDirection.NO_CHANGE
            reason = "Optimizing all strategies"
            
            for sid in strategy_ids:
                if sid in current_params:
                    optimized_params, opt_direction, opt_reason = self.optimize_strategy_params(
                        backtest_result=backtest_result,
                        current_params=current_params[sid],
                        config=config
                    )
                    optimized_params_dict[sid] = optimized_params
                    if opt_direction != OptimizationDirection.NO_CHANGE:
                        direction = opt_direction
                        reason = opt_reason
                else:
                    strategy = self.stock_selector_service.strategy_manager.get_strategy(sid)
                    if strategy:
                        optimized_params_dict[sid] = StrategyParamsSnapshot(
                            strategy_id=sid,
                            strategy_name=strategy.display_name,
                            score_multiplier=strategy.score_multiplier,
                            max_raw_score=100.0,
                            enabled=True,
                            custom_params={}
                        )
            
            performance_improvement = None
            if parent_iteration_id:
                pass
            
            iteration_record = IterationRecord(
                iteration_id=iteration_id,
                iteration_number=iteration_number,
                timestamp=datetime.now(),
                strategy_params=optimized_params_dict,
                stock_selection_result=selection_result,
                backtest_result=backtest_result,
                optimization_direction=direction,
                optimization_reason=reason,
                parent_iteration_id=parent_iteration_id,
                performance_improvement_pct=performance_improvement,
                extra_data={}
            )
            
            logger.info(f"Completed iteration {iteration_number}: return={backtest_result.total_return_pct:.2f}%, win_rate={backtest_result.win_rate_pct:.2f}%")
            return iteration_record
            
        except Exception as e:
            logger.error(f"Failed to run optimization iteration {iteration_number}: {e}")
            return None

    def run_full_optimization(
        self,
        config: OptimizationConfig,
        strategy_ids: List[str],
        start_date: date,
        top_n: int = 5,
        run_until_latest: bool = False,
    ) -> List[IterationRecord]:
        """
        Run full optimization loop.
        
        Args:
            config: Optimization configuration
            strategy_ids: List of strategy IDs to optimize
            start_date: Initial start date
            top_n: Number of top stocks to select
            run_until_latest: If True, run until latest available data instead of max_iterations
            
        Returns:
            List of iteration records
        """
        logger.info(f"Starting full optimization with {len(strategy_ids)} strategies")
        
        iteration_records: List[IterationRecord] = []
        current_params: Dict[str, StrategyParamsSnapshot] = {}
        current_date = start_date
        parent_iteration_id = None
        
        # Get latest available date if running until latest
        latest_date = None
        if run_until_latest:
            latest_date, msg = self.get_latest_available_date()
            if not latest_date:
                logger.error(f"Cannot run until latest: {msg}")
                return iteration_records
            logger.info(f"Will run until latest available date: {latest_date}")
        
        # Calculate total iterations for progress bar
        if run_until_latest and latest_date:
            days_diff = (latest_date - start_date).days
            total_iterations = max(1, days_diff // config.rebalance_frequency_days)
        else:
            total_iterations = config.max_iterations
        
        progress = ProgressBar(total=total_iterations, description="Optimization Iterations")
        
        try:
            iteration_number = 1
            while True:
                # Check if we should stop
                if not run_until_latest and iteration_number > config.max_iterations:
                    logger.info(f"Reached max iterations: {config.max_iterations}")
                    break
                
                if run_until_latest and latest_date:
                    # Check if current date is too close to latest date
                    end_date = current_date + timedelta(days=config.backtest_window_days)
                    if end_date > latest_date:
                        logger.info(f"Reached latest available date: {latest_date}")
                        break
                
                iteration_record = self.run_optimization_iteration(
                    start_date=current_date,
                    strategy_ids=strategy_ids,
                    config=config,
                    current_params=current_params,
                    iteration_number=iteration_number,
                    parent_iteration_id=parent_iteration_id,
                    top_n=top_n
                )
                
                if iteration_record:
                    iteration_records.append(iteration_record)
                    for sid, params in iteration_record.strategy_params.items():
                        current_params[sid] = params
                        self.stock_selector_service.strategy_manager.set_strategy_multiplier(
                            sid, params.score_multiplier
                        )
                    parent_iteration_id = iteration_record.iteration_id
                    
                    has_converged, message = self.check_convergence(iteration_records, config)
                    logger.info(f"Convergence check: {message}")
                    
                    if has_converged:
                        logger.info("Optimization converged, stopping early")
                        break
                        
                    current_date = current_date + timedelta(days=config.rebalance_frequency_days)
                else:
                    logger.warning(f"Iteration {iteration_number} failed, skipping")
                    break
                
                progress.update(iteration_number)
                iteration_number += 1
                    
        except Exception as e:
            logger.error(f"Full optimization failed: {e}")
        
        progress.finish(f"Completed {len(iteration_records)} optimization iterations")
        logger.info(f"Completed optimization with {len(iteration_records)} iterations")
        return iteration_records

    def save_iteration_records(
        self,
        records: List[IterationRecord],
        file_path: str,
    ) -> bool:
        """
        Save iteration records to JSON file.
        
        Args:
            records: List of iteration records
            file_path: Path to save the file
            
        Returns:
            True if successful
        """
        def _json_default(obj):
            """Custom JSON serializer for types that aren't serializable by default."""
            try:
                import numpy as np
                if isinstance(obj, (bool, np.bool_)):
                    return bool(obj)
                if isinstance(obj, (int, float, np.integer, np.floating)):
                    return float(obj)
                if isinstance(obj, (list, tuple, np.ndarray)):
                    return list(obj)
            except ImportError:
                if isinstance(obj, bool):
                    return bool(obj)
                if isinstance(obj, (int, float)):
                    return float(obj)
                if isinstance(obj, (list, tuple)):
                    return list(obj)
            
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            if isinstance(obj, dict):
                return dict(obj)
            if obj is None:
                return None
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
        
        try:
            data = [r.to_dict() for r in records]
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
            logger.info(f"Saved {len(records)} iteration records to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save iteration records: {e}")
            return False

    def update_optimization_config(self, records: List[IterationRecord], config_path: Optional[Path] = None) -> bool:
        """
        Update the optimization_config.yaml file with the latest strategy parameters from the last iteration.
        
        Args:
            records: List of iteration records
            config_path: Path to optimization_config.yaml (optional, uses default location)
            
        Returns:
            True if successful
        """
        if not records:
            logger.warning("No records to update config with")
            return False
        
        try:
            if config_path is None:
                config_path = Path(__file__).parent / "optimization_config.yaml"
            
            if not config_path.exists():
                logger.warning(f"Config file not found at {config_path}, cannot update")
                return False
            
            # Get the last iteration record
            last_record = records[-1]
            
            # Read existing config
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Update strategy parameters
            if 'strategies' not in config_data:
                config_data['strategies'] = {}
            
            updated_count = 0
            for strategy_id, params in last_record.strategy_params.items():
                # Convert to standard Python float to avoid numpy types
                score_multiplier = float(params.score_multiplier)
                
                # Try exact match first
                if strategy_id in config_data['strategies']:
                    config_data['strategies'][strategy_id]['score_multiplier'] = score_multiplier
                    updated_count += 1
                    logger.info(f"Updated {strategy_id}: score_multiplier = {score_multiplier}")
                else:
                    # Try matching without suffix (e.g., "xxx_python" -> "xxx")
                    base_id = strategy_id
                    if strategy_id.endswith('_python'):
                        base_id = strategy_id[:-7]
                    elif strategy_id.endswith('_nl'):
                        base_id = strategy_id[:-3]
                    
                    if base_id in config_data['strategies']:
                        config_data['strategies'][base_id]['score_multiplier'] = score_multiplier
                        updated_count += 1
                        logger.info(f"Updated {base_id} (matched from {strategy_id}): score_multiplier = {score_multiplier}")
                    else:
                        logger.warning(f"Strategy {strategy_id} (base: {base_id}) not found in config file")
            
            if updated_count == 0:
                logger.warning("No strategies were updated in config file")
                return False
            
            # Write updated config back to file
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False, indent=2)
            
            logger.info(f"Successfully updated config file at {config_path} with {updated_count} strategy parameters")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update optimization config: {e}", exc_info=True)
            return False

    def load_iteration_records(
        self,
        file_path: str,
    ) -> List[IterationRecord]:
        """
        Load iteration records from JSON file.
        
        Args:
            file_path: Path to load from
            
        Returns:
            List of iteration records
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            records = [IterationRecord.from_dict(d) for d in data]
            logger.info(f"Loaded {len(records)} iteration records from {file_path}")
            return records
        except Exception as e:
            logger.error(f"Failed to load iteration records: {e}")
            return []

    def get_best_iteration(
        self,
        records: List[IterationRecord],
    ) -> Optional[IterationRecord]:
        """
        Get the best iteration from records.
        
        Args:
            records: List of iteration records
            
        Returns:
            Best iteration or None
        """
        if not records:
            return None
            
        best = max(records, key=lambda r: (
            r.backtest_result.total_return_pct,
            r.backtest_result.win_rate_pct
        ))
        return best

    def print_evolution_path(
        self,
        records: List[IterationRecord],
    ) -> None:
        """
        Print evolution path as text table.
        
        Args:
            records: List of iteration records
        """
        if not records:
            print("No iteration records to display")
            return
            
        print("\n" + "=" * 140)
        header = f"{'Iteration':<10} {'Date':<20} {'Return (%)':<12} {'Win Rate (%)':<14} {'Drawdown (%)':<14}"
        
        strategy_ids = list(records[0].strategy_params.keys())
        for sid in strategy_ids:
            header += f" {sid:<14}"
        
        print(header)
        print("=" * 140)
        
        for record in records:
            bt = record.backtest_result
            line = (
                f"{record.iteration_number:<10} "
                f"{record.timestamp.strftime('%Y-%m-%d %H:%M'):<20} "
                f"{bt.total_return_pct:>10.2f}% "
                f"{bt.win_rate_pct:>12.2f}% "
                f"{bt.max_drawdown_pct:>12.2f}% "
            )
            
            for sid in strategy_ids:
                params = record.strategy_params.get(sid)
                if params:
                    line += f"{params.score_multiplier:>14.2f} "
                else:
                    line += f"{'N/A':<14} "
            
            print(line)
        
        print("=" * 140)
        print(f"\nTotal iterations: {len(records)}")

    def print_ascii_chart(
        self,
        records: List[IterationRecord],
    ) -> None:
        """
        Print simple ASCII chart.
        
        Args:
            records: List of iteration records
        """
        if not records:
            print("No records to chart")
            return
            
        print("\n" + "=" * 80)
        print("TOTAL RETURN EVOLUTION")
        print("=" * 80)
        
        returns = [r.backtest_result.total_return_pct for r in records]
        min_ret = min(returns) if returns else 0
        max_ret = max(returns) if returns else 100
        range_ret = max_ret - min_ret if max_ret != min_ret else 1
        
        chart_height = 20
        for i in range(chart_height):
            y_val = max_ret - (i / (chart_height - 1)) * range_ret
            line = f"{y_val:>6.1f}% |"
            
            for j, ret in enumerate(returns):
                ret_y = (max_ret - ret) / range_ret * (chart_height - 1)
                if abs(ret_y - i) < 0.5:
                    line += " ●"
                else:
                    line += "  "
            print(line)
            
        print(" " * 8 + "-" * (len(returns) * 2 + 1))
        print(" " * 8 + "".join(f"{i+1:2}" for i in range(len(returns))))
        print(" " * 10 + "Iterations")

    def print_best_strategy(
        self,
        records: List[IterationRecord],
    ) -> None:
        """
        Print best strategy details.
        
        Args:
            records: List of iteration records
        """
        best = self.get_best_iteration(records)
        if not best:
            print("No best strategy found")
            return
            
        print("\n" + "=" * 80)
        print("BEST ITERATION")
        print("=" * 80)
        print(f"Iteration: {best.iteration_number}")
        print(f"\nBacktest Results:")
        print(f"  Total Return: {best.backtest_result.total_return_pct:.2f}%")
        print(f"  Win Rate: {best.backtest_result.win_rate_pct:.2f}%")
        print(f"  Max Drawdown: {best.backtest_result.max_drawdown_pct:.2f}%")
        print(f"  Total Trades: {best.backtest_result.total_trades}")
        print(f"  Winning Trades: {best.backtest_result.winning_trades}")
        print(f"  Losing Trades: {best.backtest_result.losing_trades}")
        print(f"\nStrategy Parameters:")
        for strategy_id, params in best.strategy_params.items():
            print(f"  {params.strategy_name} ({strategy_id}):")
            print(f"    Score Multiplier: {params.score_multiplier:.2f}")
        print(f"\nOptimization Direction: {best.optimization_direction.value}")
        print(f"Reason: {best.optimization_reason}")
        print("=" * 80)

    def generate_walk_forward_windows(
        self,
        start_date: date,
        config: OptimizationConfig,
    ) -> List[WalkForwardWindow]:
        """
        Generate walk-forward windows for analysis.
        
        Args:
            start_date: Start date for the first window
            config: Optimization configuration
            
        Returns:
            List of WalkForwardWindow
        """
        windows = []
        
        training_days = config.wfa_training_window_days
        testing_days = config.wfa_testing_window_days
        step_days = config.wfa_step_days
        num_windows = config.wfa_num_windows
        
        current_date = start_date
        window_number = 1
        
        # Get database date range
        earliest_date, latest_date = None, None
        try:
            with self.db_manager.get_session() as session:
                from sqlalchemy import select, func
                stmt = select(
                    func.min(StockDaily.date),
                    func.max(StockDaily.date)
                )
                result = session.execute(stmt).first()
                if result:
                    earliest_date, latest_date = result
                    logger.info(f"Database date range: {earliest_date} to {latest_date}")
        except Exception as e:
            logger.warning(f"Failed to get database date range: {e}")
        
        # Get latest available date for auto-stop mode if not already known
        latest_available_date = latest_date
        if num_windows == -1 and not latest_available_date:
            latest_available_date, msg = self.get_latest_available_date()
            if not latest_available_date:
                logger.warning(f"Could not get latest available date: {msg}, defaulting to 5 windows")
                num_windows = 5
            else:
                logger.info(f"Auto-sliding mode enabled, latest available date: {latest_available_date}")
        
        while True:
            # Check if we should stop (fixed window count)
            if num_windows != -1 and window_number > num_windows:
                logger.info(f"Reached requested number of windows ({num_windows}), stopping")
                break
            
            # Calculate window dates
            training_start = current_date
            training_end = training_start + timedelta(days=training_days - 1)
            testing_start = training_end + timedelta(days=1)
            testing_end = testing_start + timedelta(days=testing_days - 1)
            
            # Check if we should stop (auto mode - no more data)
            if num_windows == -1 and latest_available_date:
                if testing_end > latest_available_date:
                    logger.info(f"Window {window_number} testing end date ({testing_end}) exceeds latest available date ({latest_available_date}), stopping")
                    break
            
            # Also check training start is not before earliest date
            if earliest_date and training_start < earliest_date:
                logger.info(f"Window {window_number} training start date ({training_start}) precedes earliest available date ({earliest_date}), stopping")
                break
            
            window_id = str(uuid.uuid4())
            
            window = WalkForwardWindow(
                window_id=window_id,
                window_number=window_number,
                training_start_date=training_start,
                training_end_date=training_end,
                testing_start_date=testing_start,
                testing_end_date=testing_end,
            )
            
            windows.append(window)
            logger.info(f"Generated window {window_number}: training={training_start} to {training_end}, testing={testing_start} to {testing_end}")
            
            current_date = current_date + timedelta(days=step_days)
            window_number += 1
        
        if not windows:
            logger.warning("No windows generated - check your start date and data availability")
        else:
            logger.info(f"Generated {len(windows)} walk-forward windows")
        
        return windows

    def run_walk_forward_analysis(
        self,
        config: OptimizationConfig,
        strategy_ids: List[str],
        start_date: date,
        top_n: int = 5,
        optimization_method: Optional[OptimizationMethod] = None,
        bayesian_n_calls: Optional[int] = None,
        bayesian_n_initial_points: Optional[int] = None,
        grid_search_step: Optional[float] = None,
        stock_codes: Optional[List[str]] = None,
    ) -> WalkForwardResult:
        """
        Run complete Walk-Forward Analysis with optimization per window.
        
        Args:
            config: Optimization configuration
            strategy_ids: List of strategy IDs to optimize
            start_date: Start date for the first window
            top_n: Number of top stocks to select
            optimization_method: Optimization method to use (Bayesian or Grid Search)
            bayesian_n_calls: Number of Bayesian optimization calls per window
            bayesian_n_initial_points: Number of initial random points for Bayesian optimization
            grid_search_step: Step size for grid search
            stock_codes: Optional list of stock codes to use (uses all stocks if not provided)
            
        Returns:
            WalkForwardResult with all windows and aggregated metrics
        """
        wfa_id = str(uuid.uuid4())
        
        # Determine optimization method
        opt_method = optimization_method if optimization_method is not None else config.optimization_method
        logger.info(f"Starting Walk-Forward Analysis (ID: {wfa_id})")
        logger.info(f"Optimization method: {opt_method.value}")
        logger.info(f"Optimization objective: {config.optimization_objective.value}")
        
        windows = self.generate_walk_forward_windows(start_date, config)
        progress = ProgressBar(total=len(windows), description="Processing windows")
        
        for i, window in enumerate(windows):
            logger.info(f"Processing window {i + 1}/{len(windows)}")
            
            if opt_method == OptimizationMethod.BAYESIAN:
                n_calls = bayesian_n_calls if bayesian_n_calls is not None else config.bayesian_n_calls
                n_initial_points = bayesian_n_initial_points if bayesian_n_initial_points is not None else config.bayesian_n_initial_points
                
                logger.info(f"Running Bayesian optimization for window {i + 1}: n_calls={n_calls}, n_initial_points={n_initial_points}")
                
                bayesian_optimizer = BayesianOptimizer(
                    optimization_engine=self,
                    config=config,
                    strategy_ids=strategy_ids,
                    start_date=window.training_start_date,
                    top_n=top_n,
                    n_initial_points=n_initial_points,
                    n_calls=n_calls,
                )
                
                best_params, bayesian_records = bayesian_optimizer.optimize()
                
                window.bayesian_optimization_records = bayesian_records
                window.best_params = best_params
                
                if bayesian_records:
                    window.training_result = bayesian_records[-1]
                    window.best_return = bayesian_optimizer.best_return
                    
                    if best_params:
                        logger.info(f"Best params found for window {i + 1}: {best_params}")
                        logger.info(f"Best return for window {i + 1}: {window.best_return:.2f}%")
                        
                        for j, strategy_id in enumerate(strategy_ids):
                            if j < len(best_params):
                                self.stock_selector_service.strategy_manager.set_strategy_multiplier(
                                    strategy_id, best_params[j]
                                )
            
            elif opt_method == OptimizationMethod.GRID_SEARCH:
                step = grid_search_step if grid_search_step is not None else config.grid_search_step
                
                logger.info(f"Running Grid Search optimization for window {i + 1}: step={step}")
                
                grid_optimizer = GridSearchOptimizer(
                    optimization_engine=self,
                    config=config,
                    strategy_ids=strategy_ids,
                    start_date=window.training_start_date,
                    top_n=top_n,
                    step=step,
                    normalize_weights=config.normalize_weights,
                    objective=config.optimization_objective,
                )
                
                best_params, grid_records = grid_optimizer.optimize()
                
                window.bayesian_optimization_records = grid_records
                window.best_params = best_params
                
                if grid_records:
                    window.training_result = grid_records[-1]
                    window.best_return = grid_optimizer.best_return
                    
                    if best_params:
                        logger.info(f"Best params found for window {i + 1}: {best_params}")
                        logger.info(f"Best return for window {i + 1}: {window.best_return:.2f}%")
                        logger.info(f"Best score for window {i + 1}: {grid_optimizer.best_score:.4f}")
                        
                        for j, strategy_id in enumerate(strategy_ids):
                            if j < len(best_params):
                                self.stock_selector_service.strategy_manager.set_strategy_multiplier(
                                    strategy_id, best_params[j]
                                )
            else:
                training_params = {}
                for sid in strategy_ids:
                    strategy = self.stock_selector_service.strategy_manager.get_strategy(sid)
                    if strategy:
                        training_params[sid] = StrategyParamsSnapshot(
                            strategy_id=sid,
                            strategy_name=strategy.display_name,
                            score_multiplier=strategy.score_multiplier,
                            max_raw_score=100.0,
                            enabled=True,
                            custom_params={},
                        )
                
                training_record = self.run_optimization_iteration(
                    start_date=window.training_start_date,
                    strategy_ids=strategy_ids,
                    config=config,
                    current_params=training_params,
                    iteration_number=i + 1,
                    top_n=top_n,
                    stock_codes=stock_codes,
                )
                
                window.training_result = training_record
                
                if training_record:
                    optimized_params = training_record.strategy_params
                    
                    for sid, params in optimized_params.items():
                        self.stock_selector_service.strategy_manager.set_strategy_multiplier(
                            sid, params.score_multiplier
                        )
            
            if window.training_result:
                logger.info(f"Finding valid testing date for window {i + 1}")
                valid_testing_date, msg = self.get_earliest_backtest_date(
                    start_date=window.testing_start_date,
                    eval_window_days=config.backtest_window_days
                )
                
                if not valid_testing_date:
                    logger.warning(f"No valid testing date found for window {i + 1}: {msg}")
                    continue
                
                logger.info(f"Using valid testing date for window {i + 1}: {valid_testing_date}")
                
                logger.info(f"Running testing selection for window {i + 1} on date: {valid_testing_date}")
                testing_selection = self.run_historical_selection(
                    target_date=valid_testing_date,
                    strategy_ids=strategy_ids,
                    top_n=top_n,
                    stock_codes=stock_codes,
                )
                
                logger.info(f"Testing selection for window {i + 1}: selected {len(testing_selection.selected_stocks)} stocks")
                
                logger.info(f"Running testing backtest for window {i + 1} on date: {valid_testing_date}, eval_window={config.backtest_window_days}")
                testing_backtest = self.run_historical_backtest(
                    selection_result=testing_selection,
                    start_date=valid_testing_date,
                    eval_window_days=config.backtest_window_days,
                )
                
                window.testing_result = testing_backtest
                
                train_ret = window.training_result.backtest_result.total_return_pct if window.training_result else 0.0
                test_ret = window.testing_result.total_return_pct if window.testing_result else 0.0
                
                logger.info(
                    f"Window {i + 1} - "
                    f"training_return={train_ret:.2f}%, "
                    f"testing_return={test_ret:.2f}%"
                )
            
            progress.update(i + 1)
        
        progress.finish()
        
        wfa_result = WalkForwardResult(
            wfa_id=wfa_id,
            windows=windows,
            config=config,
            strategy_ids=strategy_ids,
        )
        
        wfa_result.compute_aggregated_metrics()
        wfa_result.compute_parameter_stability()
        
        logger.info(f"Walk-Forward Analysis completed")
        return wfa_result

    def print_walk_forward_result(
        self,
        wfa_result: WalkForwardResult,
        detailed: bool = False,
    ) -> None:
        """
        Print Walk-Forward Analysis result.
        
        Args:
            wfa_result: WalkForwardResult to display
            detailed: Whether to show detailed Bayesian optimization information
        """
        if not wfa_result.windows:
            print("No Walk-Forward Analysis results to display")
            return
        
        print("\n" + "=" * 120)
        opt_method = wfa_result.config.optimization_method.value
        opt_obj = wfa_result.config.optimization_objective.value
        print(f"WALK-FORWARD ANALYSIS RESULTS ({opt_method.upper()} OPTIMIZATION - {opt_obj.upper()})")
        print("=" * 120)
        
        print(f"\nConfiguration:")
        print(f"  Training window: {wfa_result.config.wfa_training_window_days} days")
        print(f"  Testing window: {wfa_result.config.wfa_testing_window_days} days")
        print(f"  Step: {wfa_result.config.wfa_step_days} days")
        print(f"  Number of windows: {len(wfa_result.windows)}")
        print(f"  Optimization method: {opt_method}")
        print(f"  Optimization objective: {opt_obj}")
        
        print(f"\nWindow Results:")
        print("-" * 120)
        header = f"{'Window':<8} {'Training Period':<25} {'Testing Period':<25} {'Best Training':<18} {'Testing Return':<18} {'Iterations':<12}"
        print(header)
        print("-" * 120)
        
        for window in wfa_result.windows:
            best_train_ret = window.best_return if window.best_return is not None else 0.0
            test_ret = window.testing_result.total_return_pct if window.testing_result else 0.0
            iterations = len(window.bayesian_optimization_records) if window.bayesian_optimization_records else 0
            
            training_period = f"{window.training_start_date} to {window.training_end_date}"
            testing_period = f"{window.testing_start_date} to {window.testing_end_date}"
            
            line = (
                f"{window.window_number:<8} "
                f"{training_period:<25} "
                f"{testing_period:<25} "
                f"{best_train_ret:>14.2f}% "
                f"{test_ret:>14.2f}% "
                f"{iterations:>12}"
            )
            print(line)
        
        print("-" * 120)
        
        if detailed:
            print(f"\nBest Parameters per Window:")
            print("-" * 120)
            for window in wfa_result.windows:
                if window.best_params:
                    print(f"\nWindow {window.window_number}:")
                    for i, strategy_id in enumerate(wfa_result.strategy_ids):
                        if i < len(window.best_params):
                            print(f"  {strategy_id}: {window.best_params[i]:.4f}")
        
        if wfa_result.parameter_stability_metrics:
            print(f"\nParameter Stability Analysis:")
            print("-" * 120)
            for strategy_id, metrics in wfa_result.parameter_stability_metrics.items():
                print(f"\n{strategy_id}:")
                print(f"  Mean: {metrics['mean']:.4f}")
                print(f"  Std Dev: {metrics['std_dev']:.4f}")
                print(f"  Coeff of Variation: {metrics['coeff_variation_pct']:.2f}%")
                print(f"  Range: [{metrics['min']:.4f}, {metrics['max']:.4f}]")
                print(f"  Range Width: {metrics['range']:.4f}")
                print(f"  Windows: {metrics['num_windows']}")
        
        if wfa_result.aggregated_metrics:
            print(f"\nAggregated Metrics:")
            metrics = wfa_result.aggregated_metrics
            if 'avg_return_pct' in metrics:
                print(f"  Average Testing Return: {metrics['avg_return_pct']:.2f}%")
            if 'return_std_pct' in metrics:
                print(f"  Return Std Dev: {metrics['return_std_pct']:.2f}%")
            if 'min_return_pct' in metrics:
                print(f"  Min Testing Return: {metrics['min_return_pct']:.2f}%")
            if 'max_return_pct' in metrics:
                print(f"  Max Testing Return: {metrics['max_return_pct']:.2f}%")
            if 'positive_window_count' in metrics and 'total_window_count' in metrics:
                pos_count = metrics['positive_window_count']
                total_count = metrics['total_window_count']
                print(f"  Positive Windows: {pos_count}/{total_count} ({pos_count/total_count*100:.1f}%)")
            if 'avg_win_rate_pct' in metrics:
                print(f"  Average Win Rate: {metrics['avg_win_rate_pct']:.2f}%")
        
        print("=" * 120)

    def save_walk_forward_result(
        self,
        wfa_result: WalkForwardResult,
        file_path: str,
    ) -> bool:
        """
        Save Walk-Forward Analysis result to JSON file.
        
        Args:
            wfa_result: WalkForwardResult to save
            file_path: Path to save to
            
        Returns:
            True if successful
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(wfa_result.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Saved Walk-Forward Analysis result to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save Walk-Forward Analysis result: {e}")
            return False


class BayesianOptimizer:
    """
    Bayesian Optimizer for strategy parameter optimization.
    
    Uses scikit-optimize to find optimal strategy parameters using Gaussian Process
    regression and Expected Improvement acquisition function.
    """
    
    def __init__(
        self,
        optimization_engine: OptimizationEngine,
        config: OptimizationConfig,
        strategy_ids: List[str],
        start_date: date,
        top_n: int = 5,
        n_initial_points: int = 5,
        n_calls: int = 30,
    ):
        """
        Initialize Bayesian Optimizer.
        
        Args:
            optimization_engine: OptimizationEngine instance
            config: Optimization configuration
            strategy_ids: List of strategy IDs to optimize
            start_date: Start date for backtesting
            top_n: Number of top stocks to select
            n_initial_points: Number of initial random points
            n_calls: Total number of optimization calls
        """
        self.optimization_engine = optimization_engine
        self.config = config
        self.strategy_ids = strategy_ids
        self.start_date = start_date
        self.top_n = top_n
        self.n_initial_points = n_initial_points
        self.n_calls = n_calls
        
        self.results = []
        self.best_params = None
        self.best_return = -float('inf')
        
        logger.info(f"BayesianOptimizer initialized for {len(strategy_ids)} strategies")
    
    def _create_param_space(self) -> List:
        """
        Create parameter space for Bayesian optimization.
        
        Returns:
            List of parameter dimensions
        """
        from skopt.space import Real
        
        dimensions = []
        for i, strategy_id in enumerate(self.strategy_ids):
            dimensions.append(
                Real(
                    self.config.min_score_multiplier,
                    self.config.max_score_multiplier,
                    name=f'multiplier_{i}_{strategy_id}'
                )
            )
        return dimensions
    
    def _objective(self, x: List[float]) -> float:
        """
        Objective function to minimize (returns negative return).
        
        Args:
            x: List of score multipliers for each strategy
            
        Returns:
            Negative total return (to minimize)
        """
        try:
            # Create strategy params from x
            current_params = {}
            for i, strategy_id in enumerate(self.strategy_ids):
                strategy = self.optimization_engine.stock_selector_service.strategy_manager.get_strategy(strategy_id)
                if strategy:
                    # Ensure score_multiplier is within valid bounds
                    bounded_multiplier = max(
                        self.config.min_score_multiplier,
                        min(self.config.max_score_multiplier, x[i])
                    )
                    current_params[strategy_id] = StrategyParamsSnapshot(
                        strategy_id=strategy_id,
                        strategy_name=strategy.display_name,
                        score_multiplier=bounded_multiplier,
                        max_raw_score=100.0,
                        enabled=True,
                        custom_params={}
                    )
            
            # Run one iteration
            iteration_record = self.optimization_engine.run_optimization_iteration(
                start_date=self.start_date,
                strategy_ids=self.strategy_ids,
                config=self.config,
                current_params=current_params,
                iteration_number=len(self.results) + 1,
                top_n=self.top_n
            )
            
            if iteration_record:
                self.results.append(iteration_record)
                total_return = iteration_record.backtest_result.total_return_pct
                
                # Update best
                if total_return > self.best_return:
                    self.best_return = total_return
                    self.best_params = x.copy()
                    logger.info(f"New best found: return={total_return:.2f}%, params={x}")
                
                # Return negative for minimization
                return -total_return
            else:
                logger.warning("Iteration failed, returning high value")
                return 1000.0
                
        except Exception as e:
            logger.error(f"Error in objective function: {e}", exc_info=True)
            return 1000.0
    
    def optimize(self) -> Tuple[Optional[List[float]], List[IterationRecord]]:
        """
        Run Bayesian optimization.
        
        Returns:
            Tuple of (best_params, iteration_records)
        """
        logger.info(f"Starting Bayesian optimization with {self.n_calls} calls")
        
        try:
            from skopt import gp_minimize
            from skopt.utils import use_named_args
            
            dimensions = self._create_param_space()
            
            # Run optimization
            result = gp_minimize(
                func=self._objective,
                dimensions=dimensions,
                n_calls=self.n_calls,
                n_initial_points=self.n_initial_points,
                random_state=self.config.random_seed,
                verbose=True
            )
            
            logger.info(f"Bayesian optimization completed")
            logger.info(f"Best return: {self.best_return:.2f}%")
            logger.info(f"Best params: {self.best_params}")
            
            return self.best_params, self.results
            
        except ImportError:
            logger.error("scikit-optimize not installed. Please install with: pip install scikit-optimize")
            return None, []
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}", exc_info=True)
            return None, self.results


class GridSearchOptimizer:
    """
    Grid Search Optimizer for strategy parameter optimization.
    
    Uses systematic grid search to find optimal strategy parameters.
    Supports weight normalization (sum to 1) and multiple optimization objectives.
    """
    
    def __init__(
        self,
        optimization_engine: OptimizationEngine,
        config: OptimizationConfig,
        strategy_ids: List[str],
        start_date: date,
        top_n: int = 5,
        step: float = 0.1,
        normalize_weights: bool = True,
        objective: OptimizationObjective = OptimizationObjective.TOTAL_RETURN,
    ):
        """
        Initialize Grid Search Optimizer.
        
        Args:
            optimization_engine: OptimizationEngine instance
            config: Optimization configuration
            strategy_ids: List of strategy IDs to optimize
            start_date: Start date for backtesting
            top_n: Number of top stocks to select
            step: Grid step size (0.05 or 0.1 recommended)
            normalize_weights: Whether to normalize weights to sum to 1
            objective: Optimization objective to use
        """
        self.optimization_engine = optimization_engine
        self.config = config
        self.strategy_ids = strategy_ids
        self.start_date = start_date
        self.top_n = top_n
        self.step = step
        self.normalize_weights = normalize_weights
        self.objective = objective
        
        self.results = []
        self.best_params = None
        self.best_score = -float('inf')
        self.best_return = -float('inf')
        
        logger.info(f"GridSearchOptimizer initialized for {len(strategy_ids)} strategies")
        logger.info(f"Step: {step}, Normalize weights: {normalize_weights}, Objective: {objective.value}")
    
    def _generate_weight_candidates(self, n_strategies: int) -> List[List[float]]:
        """
        Generate weight candidates for grid search.
        
        Args:
            n_strategies: Number of strategies to generate weights for
            
        Returns:
            List of weight combinations
        """
        import numpy as np
        
        candidates = []
        weight_range = np.arange(self.config.min_score_multiplier, 1.0 + self.step / 2, self.step)
        
        if n_strategies == 1:
            return [[1.0]]
        
        if n_strategies == 2:
            for w1 in weight_range:
                w2 = 1.0 - w1
                if w2 >= -1e-6:
                    candidates.append([w1, max(self.config.min_score_multiplier, w2)])
            return candidates
        
        if n_strategies == 3:
            for w1 in weight_range:
                for w2 in weight_range:
                    w3 = 1.0 - w1 - w2
                    if w3 >= -1e-6:
                        candidates.append([w1, w2, max(self.config.min_score_multiplier, w3)])
            return candidates
        
        if self.normalize_weights:
            from itertools import product
            for weights in product(weight_range, repeat=n_strategies):
                total = sum(weights)
                if total > 1e-6:
                    normalized = [w / total for w in weights]
                    candidates.append(normalized)
        else:
            from itertools import product
            for weights in product(weight_range, repeat=n_strategies):
                candidates.append(list(weights))
        
        return candidates
    
    def _calculate_objective_score(self, backtest_result: BacktestResultSnapshot) -> float:
        """
        Calculate objective score based on selected optimization objective.
        
        Args:
            backtest_result: Backtest result snapshot
            
        Returns:
            Objective score (higher is better)
        """
        if self.objective == OptimizationObjective.TOTAL_RETURN:
            return backtest_result.total_return_pct
        
        elif self.objective == OptimizationObjective.SHARPE_RATIO:
            if backtest_result.avg_simulated_return_pct is not None:
                return backtest_result.avg_simulated_return_pct
            elif backtest_result.avg_stock_return_pct is not None:
                return backtest_result.avg_stock_return_pct
            return backtest_result.total_return_pct
        
        elif self.objective == OptimizationObjective.CALMAR_RATIO:
            if backtest_result.max_drawdown_pct and backtest_result.max_drawdown_pct > 0:
                return backtest_result.total_return_pct / backtest_result.max_drawdown_pct
            return backtest_result.total_return_pct
        
        elif self.objective == OptimizationObjective.RANK_IC:
            return backtest_result.total_return_pct
        
        return backtest_result.total_return_pct
    
    def _evaluate_weights(self, weights: List[float]) -> Tuple[Optional[float], Optional[float], Optional[IterationRecord]]:
        """
        Evaluate a single weight combination.
        
        Args:
            weights: List of weights for each strategy
            
        Returns:
            Tuple of (objective_score, total_return, iteration_record)
        """
        try:
            current_params = {}
            for i, strategy_id in enumerate(self.strategy_ids):
                strategy = self.optimization_engine.stock_selector_service.strategy_manager.get_strategy(strategy_id)
                if strategy:
                    # Ensure score_multiplier is within valid bounds
                    bounded_multiplier = max(
                        self.config.min_score_multiplier,
                        min(self.config.max_score_multiplier, weights[i])
                    )
                    current_params[strategy_id] = StrategyParamsSnapshot(
                        strategy_id=strategy_id,
                        strategy_name=strategy.display_name,
                        score_multiplier=bounded_multiplier,
                        max_raw_score=100.0,
                        enabled=True,
                        custom_params={}
                    )
            
            iteration_record = self.optimization_engine.run_optimization_iteration(
                start_date=self.start_date,
                strategy_ids=self.strategy_ids,
                config=self.config,
                current_params=current_params,
                iteration_number=len(self.results) + 1,
                top_n=self.top_n
            )
            
            if iteration_record:
                self.results.append(iteration_record)
                objective_score = self._calculate_objective_score(iteration_record.backtest_result)
                total_return = iteration_record.backtest_result.total_return_pct
                return objective_score, total_return, iteration_record
            else:
                return None, None, None
                
        except Exception as e:
            logger.error(f"Error evaluating weights {weights}: {e}", exc_info=True)
            return None, None, None
    
    def optimize(self) -> Tuple[Optional[List[float]], List[IterationRecord]]:
        """
        Run grid search optimization.
        
        Returns:
            Tuple of (best_params, iteration_records)
        """
        import numpy as np
        
        n_strategies = len(self.strategy_ids)
        candidates = self._generate_weight_candidates(n_strategies)
        
        logger.info(f"Starting grid search with {len(candidates)} weight combinations")
        
        progress = ProgressBar(total=len(candidates), description="Grid Search")
        
        for i, weights in enumerate(candidates):
            objective_score, total_return, iteration_record = self._evaluate_weights(weights)
            
            if objective_score is not None and objective_score > self.best_score:
                self.best_score = objective_score
                self.best_params = weights.copy()
                self.best_return = total_return if total_return is not None else -float('inf')
                logger.info(f"New best found: score={objective_score:.4f}, return={self.best_return:.2f}%, weights={weights}")
            
            progress.update(i + 1)
        
        progress.finish(f"Grid search completed: {len(candidates)} combinations evaluated")
        
        logger.info(f"Grid search completed")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best return: {self.best_return:.2f}%")
        logger.info(f"Best params: {self.best_params}")
        
        return self.best_params, self.results


def main():
    """
    Main CLI entry point for the optimization system.
    """
    import argparse
    import sys
    
    # First check for debug flag
    debug_mode = '--debug' in sys.argv
    if debug_mode:
        sys.argv.remove('--debug')
    
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    parser = argparse.ArgumentParser(
        description='Strategy Optimization System Command Line Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available strategies (Windows/Linux/Mac)
  python -m stock_selector.optimization_system list
  
  # Optimize a single strategy (single line)
  python -m stock_selector.optimization_system optimize --strategies volume_breakout --start-date 2025-01-01 --max-iterations 50 --save records.json
  
  # Optimize ALL strategies (use "all" keyword)
  python -m stock_selector.optimization_system optimize --strategies all --start-date 2025-01-01 --max-iterations 50 --save all_records.json
  
  # Optimize multiple specific strategies
  python -m stock_selector.optimization_system optimize --strategies volume_breakout ma_golden_cross short_term --start-date 2025-01-01 --max-iterations 50 --save records.json
  
  # Windows PowerShell (use backtick ` for line continuation)
  python -m stock_selector.optimization_system optimize `
      --strategies all `
      --start-date 2025-01-01 `
      --max-iterations 50 `
      --save all_records.json
  
  # Linux/Mac (use backslash \\ for line continuation)
  python -m stock_selector.optimization_system optimize \\
      --strategies all \\
      --start-date 2025-01-01 \\
      --max-iterations 50 \\
      --save all_records.json
  
  # Load and display records
  python -m stock_selector.optimization_system load --file records.json
  
  # Debug mode for troubleshooting
  python -m stock_selector.optimization_system walk-forward --debug --strategies all --num-windows 1
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    list_parser = subparsers.add_parser('list', help='List all available strategies')
    
    earliest_date_parser = subparsers.add_parser('earliest-date', help='Find earliest date that can be used for backtesting')
    earliest_date_parser.add_argument('--eval-window-days', type=int, default=30, help='Evaluation window days (default: 30)')
    
    optimize_parser = subparsers.add_parser('optimize', help='Run strategy optimization')
    optimize_parser.add_argument('--strategies', nargs='+', help='Strategy IDs to optimize, or "all" to optimize all strategies')
    optimize_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD), or use "earliest" to auto-detect (default: earliest)')
    optimize_parser.add_argument('--max-iterations', type=int, help='Maximum iterations (if not set, run until latest data)')
    optimize_parser.add_argument('--top-n', type=int, default=5, help='Top N stocks to select (default: 5)')
    optimize_parser.add_argument('--save', default='records.json', help='Save records to JSON file (default: records.json)')
    optimize_parser.add_argument('--update-config', action='store_true', help='Update optimization_config.yaml with optimized parameters after backtest')
    
    bayesian_parser = subparsers.add_parser('bayesian-optimize', help='Run Bayesian optimization for strategy parameters')
    bayesian_parser.add_argument('--strategies', nargs='+', help='Strategy IDs to optimize, or "all" to optimize all strategies')
    bayesian_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD), or use "earliest" to auto-detect (default: earliest)')
    bayesian_parser.add_argument('--n-calls', type=int, help='Number of Bayesian optimization calls (uses config file default if not specified)')
    bayesian_parser.add_argument('--n-initial-points', type=int, help='Number of initial random points (uses config file default if not specified)')
    bayesian_parser.add_argument('--top-n', type=int, default=5, help='Top N stocks to select (default: 5)')
    bayesian_parser.add_argument('--save', default='bayesian_records.json', help='Save records to JSON file (default: bayesian_records.json)')
    bayesian_parser.add_argument('--update-config', action='store_true', help='Update optimization_config.yaml with optimized parameters')
    bayesian_parser.add_argument('--walk-forward', action='store_true', help='Run Walk-Forward Analysis after optimization to validate results')
    bayesian_parser.add_argument('--save-wfa', default='bayesian_wfa_result.json', help='Save WFA result to JSON file (default: bayesian_wfa_result.json)')
    
    wfa_parser = subparsers.add_parser('walk-forward', help='Run Walk-Forward Analysis for strategy validation')
    wfa_parser.add_argument('--strategies', nargs='+', help='Strategy IDs to test, or "all" to test all strategies')
    wfa_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD), or use "earliest" to auto-detect (default: earliest)')
    wfa_parser.add_argument('--training-window', type=int, help='Training window days (uses config file default if not specified)')
    wfa_parser.add_argument('--testing-window', type=int, help='Testing window days (uses config file default if not specified)')
    wfa_parser.add_argument('--step', type=int, help='Step days between windows (uses config file default if not specified)')
    wfa_parser.add_argument('--num-windows', type=int, help='Number of walk-forward windows (set to -1 for auto until no more data, uses config file default if not specified)')
    wfa_parser.add_argument('--top-n', type=int, default=5, help='Top N stocks to select (default: 5)')
    wfa_parser.add_argument('--save', default='walk_forward_result.json', help='Save WFA result to JSON file (default: walk_forward_result.json)')
    wfa_parser.add_argument('--optimization-method', choices=['bayesian', 'grid_search'], help='Optimization method: bayesian or grid_search (uses config file default if not specified)')
    wfa_parser.add_argument('--objective', choices=['total_return', 'sharpe_ratio', 'calmar_ratio', 'rank_ic'], help='Optimization objective (uses config file default if not specified)')
    wfa_parser.add_argument('--bayesian-n-calls', type=int, help='Number of Bayesian optimization calls per window (uses config default)')
    wfa_parser.add_argument('--bayesian-n-initial-points', type=int, help='Number of initial random points for Bayesian optimization (uses config default)')
    wfa_parser.add_argument('--grid-search-step', type=float, help='Step size for grid search (uses config default)')
    wfa_parser.add_argument('--detailed', action='store_true', help='Show detailed optimization information')
    wfa_parser.add_argument('--update-config', action='store_true', help='Update optimization_config.yaml with averaged best parameters from all windows')
    
    load_parser = subparsers.add_parser('load', help='Load and display optimization records')
    load_parser.add_argument('--file', required=True, help='JSON file path to load')
    
    args = parser.parse_args()
    
    try:
        from data_provider import DataFetcherManager
        data_fetcher_manager = DataFetcherManager()
        
        from stock_selector import StockSelectorService
        service = StockSelectorService()
        service.set_data_provider(data_fetcher_manager)
        
        engine = OptimizationEngine(stock_selector_service=service)
        
        if args.command == 'list':
            strategies = service.get_available_strategies()
            print("\nAvailable Strategies:")
            print("=" * 80)
            for strategy in strategies:
                print(f"\n{strategy.display_name}")
                print(f"  ID: {strategy.id}")
                print(f"  Type: {strategy.strategy_type.name}")
                print(f"  Description: {strategy.description}")
            print("\n" + "=" * 80)
            print(f"Total: {len(strategies)} strategies")
            
        elif args.command == 'earliest-date':
            earliest_date, message = engine.get_earliest_backtest_date(
                eval_window_days=args.eval_window_days
            )
            if earliest_date:
                print(f"\n{message}")
                print(f"Earliest backtest date: {earliest_date}")
            else:
                print(f"\n{message}")
                return 1
            
        elif args.command == 'optimize':
            from datetime import datetime
            
            # Handle start date
            if not args.start_date or args.start_date.lower() == 'earliest':
                start_date, message = engine.get_earliest_backtest_date(eval_window_days=30)
                if not start_date:
                    print(f"Error: {message}")
                    return 1
                print(f"Using earliest available date: {start_date}")
            else:
                start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
            
            # Handle "all" strategy option
            strategy_ids = args.strategies
            if strategy_ids and len(strategy_ids) == 1 and strategy_ids[0].lower() == 'all':
                all_strategies = service.get_available_strategies()
                strategy_ids = [s.id for s in all_strategies]
                print(f"Using all {len(strategy_ids)} available strategies")
            
            config = OptimizationConfig.from_yaml()
            run_until_latest = False
            if args.max_iterations:
                config.max_iterations = args.max_iterations
                print(f"Max iterations: {args.max_iterations}")
            else:
                run_until_latest = True
                print("Running until latest available data")
            
            print(f"Starting optimization for strategies: {strategy_ids}")
            print(f"Start date: {start_date}")
            
            records = engine.run_full_optimization(
                config=config,
                strategy_ids=strategy_ids,
                start_date=start_date,
                top_n=args.top_n,
                run_until_latest=run_until_latest
            )
            
            if records:
                print("\nOptimization completed!")
                engine.print_evolution_path(records)
                engine.print_ascii_chart(records)
                engine.print_best_strategy(records)
                
                if engine.save_iteration_records(records, args.save):
                    print(f"\nRecords saved to: {args.save}")
                
                if args.update_config:
                    print("\nUpdating optimization_config.yaml with optimized parameters...")
                    if engine.update_optimization_config(records):
                        print("✓ optimization_config.yaml updated successfully!")
                    else:
                        print("✗ Failed to update optimization_config.yaml")
            else:
                print("No iteration records generated.")
                
        elif args.command == 'bayesian-optimize':
            from datetime import datetime
            
            # Handle start date
            if not args.start_date or args.start_date.lower() == 'earliest':
                start_date, message = engine.get_earliest_backtest_date(eval_window_days=30)
                if not start_date:
                    print(f"Error: {message}")
                    return 1
                print(f"Using earliest available date: {start_date}")
            else:
                start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
            
            # Handle "all" strategy option
            strategy_ids = args.strategies
            if strategy_ids and len(strategy_ids) == 1 and strategy_ids[0].lower() == 'all':
                all_strategies = service.get_available_strategies()
                strategy_ids = [s.id for s in all_strategies]
                print(f"Using all {len(strategy_ids)} available strategies")
            
            config = OptimizationConfig.from_yaml()
            print(f"Starting Bayesian optimization for strategies: {strategy_ids}")
            print(f"Start date: {start_date}")
            
            # Use config file defaults if not specified in command line
            n_calls = args.n_calls if args.n_calls is not None else config.bayesian_n_calls
            n_initial_points = args.n_initial_points if args.n_initial_points is not None else config.bayesian_n_initial_points
            
            print(f"Number of calls: {n_calls}")
            print(f"Number of initial points: {n_initial_points}")
            
            # Create and run Bayesian optimizer
            bayesian_optimizer = BayesianOptimizer(
                optimization_engine=engine,
                config=config,
                strategy_ids=strategy_ids,
                start_date=start_date,
                top_n=args.top_n,
                n_initial_points=n_initial_points,
                n_calls=n_calls
            )
            
            best_params, records = bayesian_optimizer.optimize()
            
            if records:
                print("\nBayesian optimization completed!")
                engine.print_evolution_path(records)
                engine.print_ascii_chart(records)
                engine.print_best_strategy(records)
                
                if best_params:
                    print("\nBest parameters found:")
                    for i, strategy_id in enumerate(strategy_ids):
                        print(f"  {strategy_id}: {best_params[i]:.4f}")
                
                if engine.save_iteration_records(records, args.save):
                    print(f"\nRecords saved to: {args.save}")
                
                if args.update_config and best_params:
                    print("\nUpdating optimization_config.yaml with optimized parameters...")
                    # Create a mock iteration record with best params for config update
                    if records:
                        best_record = records[-1]
                        # Update with best params
                        for i, strategy_id in enumerate(strategy_ids):
                            if strategy_id in best_record.strategy_params:
                                best_record.strategy_params[strategy_id].score_multiplier = best_params[i]
                        if engine.update_optimization_config([best_record]):
                            print("✓ optimization_config.yaml updated successfully!")
                        else:
                            print("✗ Failed to update optimization_config.yaml")
                
                # Run Walk-Forward Analysis if requested
                if args.walk_forward and best_params:
                    print("\n" + "=" * 100)
                    print("Running Walk-Forward Analysis to validate results...")
                    print("=" * 100)
                    
                    # Apply best parameters to strategies
                    for i, strategy_id in enumerate(strategy_ids):
                        service.strategy_manager.set_strategy_multiplier(strategy_id, best_params[i])
                    
                    # Run WFA
                    wfa_result = engine.run_walk_forward_analysis(
                        config=config,
                        strategy_ids=strategy_ids,
                        start_date=start_date,
                        top_n=args.top_n,
                    )
                    
                    if wfa_result:
                        engine.print_walk_forward_result(wfa_result)
                        if engine.save_walk_forward_result(wfa_result, args.save_wfa):
                            print(f"\nWFA validation result saved to: {args.save_wfa}")
                    else:
                        print("No Walk-Forward Analysis result generated.")
            else:
                print("No iteration records generated.")
                
        elif args.command == 'walk-forward':
            from datetime import datetime
            
            config = OptimizationConfig.from_yaml()
            
            # Priority: command line argument > config file > default (2021-01-01)
            if args.start_date and args.start_date.lower() != 'earliest':
                # Use command line specified date
                start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
                print(f"Using command line specified date: {start_date}")
            elif config.wfa_start_date:
                # Use config file date
                start_date = config.wfa_start_date
                print(f"Using config file date: {start_date}")
            else:
                # Fallback to finding earliest available date
                start_date, message = engine.get_earliest_backtest_date(eval_window_days=30)
                if not start_date:
                    print(f"Error: {message}")
                    return 1
                print(f"Using earliest available date: {start_date}")
            
            strategy_ids = args.strategies
            if strategy_ids and len(strategy_ids) == 1 and strategy_ids[0].lower() == 'all':
                all_strategies = service.get_available_strategies()
                strategy_ids = [s.id for s in all_strategies]
                print(f"Using all {len(strategy_ids)} available strategies")
            
            if args.training_window is not None:
                config.wfa_training_window_days = args.training_window
            if args.testing_window is not None:
                config.wfa_testing_window_days = args.testing_window
            if args.step is not None:
                config.wfa_step_days = args.step
            if args.num_windows is not None:
                config.wfa_num_windows = args.num_windows
            
            print(f"Starting Walk-Forward Analysis for strategies: {strategy_ids}")
            print(f"Start date: {start_date}")
            print(f"Training window: {config.wfa_training_window_days} days")
            print(f"Testing window: {config.wfa_testing_window_days} days")
            print(f"Step: {config.wfa_step_days} days")
            print(f"Number of windows: {config.wfa_num_windows}")
            
            # Parse optimization method
            optimization_method = None
            if args.optimization_method:
                optimization_method = OptimizationMethod(args.optimization_method)
                print(f"Optimization method: {optimization_method.value}")
            
            # Parse optimization objective
            if args.objective:
                config.optimization_objective = OptimizationObjective(args.objective)
                print(f"Optimization objective: {config.optimization_objective.value}")
            
            # Print optimization specific parameters
            opt_method = optimization_method if optimization_method else config.optimization_method
            if opt_method == OptimizationMethod.BAYESIAN:
                if args.bayesian_n_calls:
                    print(f"Bayesian n_calls: {args.bayesian_n_calls}")
                if args.bayesian_n_initial_points:
                    print(f"Bayesian n_initial_points: {args.bayesian_n_initial_points}")
            elif opt_method == OptimizationMethod.GRID_SEARCH:
                if args.grid_search_step:
                    print(f"Grid search step: {args.grid_search_step}")
                print(f"Normalize weights: {config.normalize_weights}")
            
            # Prepare stock data before starting WFA
            print("\n" + "=" * 120)
            print("PREPARING STOCK DATA FOR WALK-FORWARD ANALYSIS")
            print("=" * 120)
            
            from stock_selector.data_validator import prepare_data_for_wfa, filter_stocks_for_wfa
            from stock_selector.stock_pool import get_all_stock_codes
            
            # Get all stock codes
            all_stock_codes = get_all_stock_codes()
            print(f"Total stock codes: {len(all_stock_codes)}")
            
            # Prepare data
            prepare_data_for_wfa(
                stock_codes=all_stock_codes,
                wfa_start_date=start_date,
                training_window_days=config.wfa_training_window_days,
                testing_window_days=config.wfa_testing_window_days,
                num_windows=config.wfa_num_windows,
                step_days=config.wfa_step_days
            )
            
            print("\nData preparation complete!")
            print("=" * 120)
            
            # Filter stocks with complete data for WFA
            print("\n" + "=" * 120)
            print("FILTERING STOCKS WITH COMPLETE DATA")
            print("=" * 120)
            
            filtered_stock_codes, filter_stats = filter_stocks_for_wfa(
                stock_codes=all_stock_codes,
                wfa_start_date=start_date,
                training_window_days=config.wfa_training_window_days,
                testing_window_days=config.wfa_testing_window_days,
                num_windows=config.wfa_num_windows,
                step_days=config.wfa_step_days,
                min_data_ratio=0.8
            )
            
            print(f"\nStock filtering complete!")
            print(f"Stocks with complete data: {filter_stats['stocks_passed']}/{filter_stats['total_stocks']}")
            print(f"Pass rate: {filter_stats['stocks_passed']/filter_stats['total_stocks']*100:.1f}%")
            print("=" * 120 + "\n")
            
            # Update stock pool to use only filtered stocks
            if filtered_stock_codes:
                print(f"Using {len(filtered_stock_codes)} stocks will be used for WFA")
            else:
                print("WARNING: No stocks passed the filtering, using all stocks anyway")
                filtered_stock_codes = all_stock_codes
            
            wfa_result = engine.run_walk_forward_analysis(
                config=config,
                strategy_ids=strategy_ids,
                start_date=start_date,
                top_n=args.top_n,
                optimization_method=optimization_method,
                bayesian_n_calls=args.bayesian_n_calls,
                bayesian_n_initial_points=args.bayesian_n_initial_points,
                grid_search_step=args.grid_search_step,
                stock_codes=filtered_stock_codes,
            )
            
            if wfa_result:
                print("\nWalk-Forward Analysis completed!")
                engine.print_walk_forward_result(wfa_result, detailed=args.detailed)
                
                if engine.save_walk_forward_result(wfa_result, args.save):
                    print(f"\nWFA result saved to: {args.save}")
                
                # Update config with averaged best parameters if requested
                if args.update_config:
                    print("\nUpdating optimization_config.yaml with averaged best parameters...")
                    
                    # Calculate average best parameters across all windows
                    windows_with_params = [w for w in wfa_result.windows if w.best_params is not None]
                    if windows_with_params:
                        num_strategies = len(wfa_result.strategy_ids)
                        avg_params = []
                        
                        for i in range(num_strategies):
                            param_values = []
                            for window in windows_with_params:
                                if window.best_params and i < len(window.best_params):
                                    param_values.append(window.best_params[i])
                            
                            if param_values:
                                avg_param = sum(param_values) / len(param_values)
                                avg_param = float(avg_param)
                                avg_params.append(avg_param)
                                print(f"  {wfa_result.strategy_ids[i]}: {avg_param:.4f} (from {len(param_values)} windows)")
                        
                        # Create a mock iteration record with averaged parameters for config update
                        if avg_params:
                            # Create strategy params dict
                            strategy_params_dict = {}
                            for i, strategy_id in enumerate(wfa_result.strategy_ids):
                                if i < len(avg_params):
                                    strategy = service.strategy_manager.get_strategy(strategy_id)
                                    if strategy:
                                        strategy_params_dict[strategy_id] = StrategyParamsSnapshot(
                                            strategy_id=strategy_id,
                                            strategy_name=strategy.display_name,
                                            score_multiplier=float(avg_params[i]),
                                            max_raw_score=100.0,
                                            enabled=True,
                                            custom_params={}
                                        )
                            
                            # Create a mock iteration record
                            if strategy_params_dict:
                                # We need at least a minimal IterationRecord
                                # Create a dummy one for the config update
                                from datetime import datetime
                                mock_record = IterationRecord(
                                    iteration_id=str(uuid.uuid4()),
                                    iteration_number=1,
                                    timestamp=datetime.now(),
                                    strategy_params=strategy_params_dict,
                                    stock_selection_result=StockSelectionResult(
                                        selection_date=datetime.now(),
                                        selected_stocks=[],
                                        total_candidates=0,
                                        min_match_score=0.0
                                    ),
                                    backtest_result=BacktestResultSnapshot(
                                        backtest_id=str(uuid.uuid4()),
                                        start_date=datetime.now(),
                                        end_date=datetime.now()
                                    ),
                                    optimization_direction=OptimizationDirection.NO_CHANGE
                                )
                                
                                if engine.update_optimization_config([mock_record]):
                                    print("✓ optimization_config.yaml updated successfully with averaged parameters!")
                                else:
                                    print("✗ Failed to update optimization_config.yaml")
                    else:
                        print("No valid parameters found across windows to update config")
            else:
                print("No Walk-Forward Analysis result generated.")
                
        elif args.command == 'load':
            records = engine.load_iteration_records(args.file)
            if records:
                print(f"Loaded {len(records)} records from {args.file}")
                engine.print_evolution_path(records)
                engine.print_ascii_chart(records)
                engine.print_best_strategy(records)
            else:
                print(f"No records loaded from {args.file}")
                
        else:
            parser.print_help()
            return 1
            
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
