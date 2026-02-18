import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from logger import log

@dataclass
class Trade:
    date: datetime
    ts_code: str
    action: str
    price: float
    shares: int
    commission: float
    total: float

@dataclass
class Position:
    ts_code: str
    shares: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

@dataclass
class Portfolio:
    cash: float = 1000000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    total_value: float = 1000000.0

class BacktestEngine:
    def __init__(self, initial_cash: float = 1000000.0, commission_rate: float = 0.0003, 
                 min_commission: float = 5.0, slippage: float = 0.001):
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage = slippage
        
        self.portfolio = Portfolio(cash=initial_cash)
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        
    def reset(self):
        self.portfolio = Portfolio(cash=self.initial_cash)
        self.trades = []
        self.equity_curve = []
        
    def run(self, df: pd.DataFrame, ts_code: str = None) -> Dict[str, Any]:
        if df.empty:
            log.error("回测数据为空")
            return {}
        
        df = df.copy().sort_values('trade_date').reset_index(drop=True)
        
        if 'position' not in df.columns:
            log.warning("数据中未找到position列，将使用signal列生成position")
            if 'signal' in df.columns:
                df['position'] = df['signal'].shift(1).fillna(0)
            else:
                log.error("数据中既没有position也没有signal列")
                return {}
        
        log.info(f"开始回测，数据条数: {len(df)}")
        
        for idx, row in df.iterrows():
            self._process_bar(row, ts_code)
        
        results = self._calculate_results(df)
        log.info("回测完成")
        
        return results
    
    def _process_bar(self, row: pd.Series, ts_code: str = None):
        date = row['trade_date']
        close = row['close']
        target_position = row.get('position', 0)
        
        code = ts_code if ts_code else row.get('ts_code', 'UNKNOWN')
        
        self._update_portfolio_values(close, code)
        current_position = self.portfolio.positions.get(code, Position(ts_code=code))
        
        if target_position > 0 and current_position.shares == 0:
            self._buy(date, code, close, target_position)
        elif target_position <= 0 and current_position.shares > 0:
            self._sell(date, code, close)
        
        self._record_equity(date)
    
    def _buy(self, date: datetime, ts_code: str, price: float, position_scale: float):
        buy_price = price * (1 + self.slippage)
        max_shares = int(self.portfolio.cash / (buy_price * 100)) * 100
        
        if max_shares <= 0:
            return
        
        shares = max_shares
        value = shares * buy_price
        commission = max(value * self.commission_rate, self.min_commission)
        total = value + commission
        
        if total > self.portfolio.cash:
            shares = int((self.portfolio.cash - self.min_commission) / (buy_price * 100)) * 100
            if shares <= 0:
                return
            value = shares * buy_price
            commission = max(value * self.commission_rate, self.min_commission)
            total = value + commission
        
        self.portfolio.cash -= total
        
        if ts_code in self.portfolio.positions:
            pos = self.portfolio.positions[ts_code]
            total_shares = pos.shares + shares
            total_cost = pos.shares * pos.avg_cost + value
            pos.shares = total_shares
            pos.avg_cost = total_cost / total_shares
        else:
            self.portfolio.positions[ts_code] = Position(
                ts_code=ts_code,
                shares=shares,
                avg_cost=buy_price,
                current_price=price
            )
        
        self.trades.append(Trade(
            date=date,
            ts_code=ts_code,
            action='BUY',
            price=buy_price,
            shares=shares,
            commission=commission,
            total=total
        ))
        
        log.debug(f"{date} 买入 {ts_code} {shares}股，价格{buy_price:.2f}，总花费{total:.2f}")
    
    def _sell(self, date: datetime, ts_code: str, price: float):
        if ts_code not in self.portfolio.positions:
            return
        
        pos = self.portfolio.positions[ts_code]
        shares = pos.shares
        
        if shares <= 0:
            return
        
        sell_price = price * (1 - self.slippage)
        value = shares * sell_price
        commission = max(value * self.commission_rate, self.min_commission)
        total = value - commission
        
        self.portfolio.cash += total
        del self.portfolio.positions[ts_code]
        
        self.trades.append(Trade(
            date=date,
            ts_code=ts_code,
            action='SELL',
            price=sell_price,
            shares=shares,
            commission=commission,
            total=total
        ))
        
        log.debug(f"{date} 卖出 {ts_code} {shares}股，价格{sell_price:.2f}，总收入{total:.2f}")
    
    def _update_portfolio_values(self, current_price: float, ts_code: str):
        total_position_value = 0.0
        
        for code, pos in self.portfolio.positions.items():
            if code == ts_code:
                pos.current_price = current_price
            pos.unrealized_pnl = (pos.current_price - pos.avg_cost) * pos.shares
            total_position_value += pos.current_price * pos.shares
        
        self.portfolio.total_value = self.portfolio.cash + total_position_value
    
    def _record_equity(self, date: datetime):
        self.equity_curve.append({
            'date': date,
            'cash': self.portfolio.cash,
            'total_value': self.portfolio.total_value,
            'positions_count': len(self.portfolio.positions)
        })
    
    def _calculate_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        equity_df = pd.DataFrame(self.equity_curve)
        
        if equity_df.empty:
            return {}
        
        equity_df['returns'] = equity_df['total_value'].pct_change().fillna(0)
        
        if 'close' in df.columns:
            benchmark_df = df[['trade_date', 'close']].copy()
            benchmark_df = benchmark_df.set_index('trade_date')
            equity_df = equity_df.set_index('date')
            merged = equity_df.join(benchmark_df, how='left')
            merged['benchmark_returns'] = merged['close'].pct_change().fillna(0)
        else:
            merged = equity_df.set_index('date')
            merged['benchmark_returns'] = 0
        
        total_return = (merged['total_value'].iloc[-1] - self.initial_cash) / self.initial_cash
        
        years = len(merged) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        daily_returns = merged['returns']
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
        
        rolling_max = merged['total_value'].cummax()
        drawdown = (merged['total_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        win_trades = 0
        total_trades = 0
        
        if self.trades:
            buy_trades = {}
            completed_trades = []
            
            for trade in self.trades:
                if trade.action == 'BUY':
                    if trade.ts_code not in buy_trades:
                        buy_trades[trade.ts_code] = []
                    buy_trades[trade.ts_code].append(trade)
                elif trade.action == 'SELL':
                    if trade.ts_code in buy_trades and buy_trades[trade.ts_code]:
                        buy_trade = buy_trades[trade.ts_code].pop(0)
                        sell_value = trade.total
                        buy_value = buy_trade.total
                        profit = sell_value - buy_value
                        
                        completed_trades.append({
                            'buy': buy_trade,
                            'sell': trade,
                            'profit': profit
                        })
                        
                        if profit > 0:
                            win_trades += 1
            
            total_trades = len(completed_trades)
        
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        results = {
            'initial_cash': self.initial_cash,
            'final_value': merged['total_value'].iloc[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'equity_curve': equity_df.to_dict('records'),
            'trades': [t.__dict__ for t in self.trades]
        }
        
        return results

backtest_engine = BacktestEngine()
