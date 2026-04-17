# -*- coding: utf-8 -*-
"""
风险监控模块测试脚本
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from watchdog.risk_manager import (
    RiskManager,
    RiskLevel,
    PositionStatus,
)


def test_basic_operations():
    """测试基本持仓操作"""
    print("=" * 80)
    print("测试 1: 基本持仓操作")
    print("=" * 80)
    
    # 创建风险管理器
    risk_mgr = RiskManager(initial_capital=1000000.0)
    print(f"\n初始资金: {risk_mgr.initial_capital:,.2f}")
    print(f"可用资金: {risk_mgr.cash_available:,.2f}")
    
    # 开仓 - 贵州茅台
    position1 = risk_mgr.open_position(
        stock_code="600519",
        stock_name="贵州茅台",
        quantity=100,
        entry_price=1800.0,
        stop_loss_price=1700.0,
        take_profit_price=1950.0,
    )
    
    assert position1 is not None, "开仓失败"
    print(f"\n✅ 开仓成功: {position1.stock_name} {position1.quantity}股 @ {position1.entry_price}")
    print(f"   持仓市值: {position1.position_value:,.2f}")
    print(f"   止损价: {position1.stop_loss_price}")
    print(f"   止盈价: {position1.take_profit_price}")
    print(f"   可用资金: {risk_mgr.cash_available:,.2f}")
    
    # 开仓 - 平安银行
    position2 = risk_mgr.open_position(
        stock_code="000001",
        stock_name="平安银行",
        quantity=500,
        entry_price=12.5,
        stop_loss_price=11.5,
        take_profit_price=14.0,
    )
    
    assert position2 is not None, "开仓失败"
    print(f"\n✅ 开仓成功: {position2.stock_name} {position2.quantity}股 @ {position2.entry_price}")
    print(f"   持仓市值: {position2.position_value:,.2f}")
    print(f"   可用资金: {risk_mgr.cash_available:,.2f}")
    
    # 获取开仓持仓
    open_positions = risk_mgr.get_open_positions()
    print(f"\n✅ 当前开仓数量: {len(open_positions)}")
    
    print("\n✅ 基本持仓操作测试通过!")


def test_price_updates_and_stops():
    """测试价格更新和止损止盈"""
    print("\n" + "=" * 80)
    print("测试 2: 价格更新和止损止盈")
    print("=" * 80)
    
    risk_mgr = RiskManager(initial_capital=1000000.0)
    
    # 开仓
    position = risk_mgr.open_position(
        stock_code="600519",
        stock_name="贵州茅台",
        quantity=100,
        entry_price=1800.0,
        stop_loss_price=1700.0,
        take_profit_price=1950.0,
    )
    
    assert position is not None
    
    # 更新价格 - 小幅上涨
    alert = risk_mgr.update_position_price(position.position_id, 1820.0)
    assert alert is None, "不应该触发预警"
    print(f"\n✅ 价格更新到 1820.0，浮动盈亏: {position.unrealized_pnl_pct:.2f}%")
    
    # 更新价格 - 触发止盈
    alert = risk_mgr.update_position_price(position.position_id, 1960.0)
    assert alert is not None, "应该触发止盈预警"
    assert alert.alert_type == "take_profit_triggered"
    print(f"✅ 触发止盈预警: {alert.message}")
    print(f"   风险级别: {alert.risk_level.value}")
    
    # 重置价格
    risk_mgr.update_position_price(position.position_id, 1800.0)
    
    # 更新价格 - 触发止损
    alert = risk_mgr.update_position_price(position.position_id, 1690.0)
    assert alert is not None, "应该触发止损预警"
    assert alert.alert_type == "stop_loss_triggered"
    print(f"✅ 触发止损预警: {alert.message}")
    print(f"   风险级别: {alert.risk_level.value}")
    
    print("\n✅ 止损止盈测试通过!")


def test_closing_positions():
    """测试平仓"""
    print("\n" + "=" * 80)
    print("测试 3: 平仓")
    print("=" * 80)
    
    risk_mgr = RiskManager(initial_capital=1000000.0)
    
    # 开仓
    position = risk_mgr.open_position(
        stock_code="600519",
        stock_name="贵州茅台",
        quantity=100,
        entry_price=1800.0,
    )
    
    assert position is not None
    initial_cash = risk_mgr.cash_available
    
    # 平仓 - 盈利
    closed_position = risk_mgr.close_position(
        position.position_id,
        close_price=1900.0,
        close_reason="take_profit",
    )
    
    assert closed_position is not None
    assert closed_position.status == PositionStatus.TAKE_PROFIT
    print(f"\n✅ 平仓成功: {closed_position.stock_name}")
    print(f"   入场价: {closed_position.entry_price}")
    print(f"   平仓价: {closed_position.close_date}")
    print(f"   实现盈亏: {closed_position.realized_pnl:,.2f}")
    print(f"   实现盈亏率: {closed_position.realized_pnl_pct:.2f}%")
    print(f"   可用资金: {risk_mgr.cash_available:,.2f}")
    
    assert risk_mgr.cash_available > initial_cash, "盈利平仓后现金应该增加"
    
    print("\n✅ 平仓测试通过!")


def test_portfolio_metrics():
    """测试组合风险指标"""
    print("\n" + "=" * 80)
    print("测试 4: 组合风险指标")
    print("=" * 80)
    
    risk_mgr = RiskManager(initial_capital=1000000.0)
    
    # 开多个仓位
    stocks = [
        ("600519", "贵州茅台", 100, 1800.0),
        ("000001", "平安银行", 500, 12.5),
        ("601318", "中国平安", 200, 45.0),
        ("000858", "五粮液", 150, 160.0),
    ]
    
    for code, name, qty, price in stocks:
        pos = risk_mgr.open_position(
            stock_code=code,
            stock_name=name,
            quantity=qty,
            entry_price=price,
        )
        assert pos is not None
    
    # 计算组合指标
    metrics = risk_mgr.calculate_portfolio_metrics()
    
    print(f"\n✅ 组合风险指标:")
    print(f"   总市值: {metrics.total_value:,.2f}")
    print(f"   总成本: {metrics.total_cost:,.2f}")
    print(f"   总盈亏: {metrics.total_pnl:,.2f}")
    print(f"   总盈亏率: {metrics.total_pnl_pct:.2f}%")
    print(f"   可用现金: {metrics.cash_available:,.2f}")
    print(f"   持仓数量: {metrics.position_count}")
    print(f"   开仓数量: {metrics.open_position_count}")
    print(f"   最大回撤: {metrics.max_drawdown:.2f}%")
    print(f"   VaR(95%): {metrics.var_95:.2f}%")
    print(f"   VaR(99%): {metrics.var_99:.2f}%")
    print(f"   杠杆率: {metrics.leverage_ratio:.2f}x")
    print(f"   集中度: {metrics.concentration_ratio:.2f}%")
    print(f"   风险级别: {metrics.risk_level.value}")
    
    # 记录净值快照
    risk_mgr.record_equity_snapshot()
    print(f"\n✅ 净值快照已记录，当前净值曲线长度: {len(risk_mgr.equity_curve)}")
    
    print("\n✅ 组合风险指标测试通过!")


def test_risk_alerts():
    """测试风险预警"""
    print("\n" + "=" * 80)
    print("测试 5: 风险预警")
    print("=" * 80)
    
    risk_mgr = RiskManager(initial_capital=1000000.0, max_position_pct=0.3)
    
    # 开一个较大仓位（高集中度）
    position = risk_mgr.open_position(
        stock_code="600519",
        stock_name="贵州茅台",
        quantity=150,  # 150股，高集中度
        entry_price=1800.0,
    )
    
    assert position is not None, "开仓应该成功"
    
    # 再开几个小仓位
    risk_mgr.open_position("000001", "平安银行", 500, 12.5)
    risk_mgr.open_position("601318", "中国平安", 200, 45.0)
    
    # 更新第一个仓位的价格，制造回撤
    risk_mgr.update_position_price(position.position_id, 1650.0)
    
    # 检查组合风险
    alerts = risk_mgr.check_portfolio_risk()
    print(f"\n✅ 组合风险检查完成，触发 {len(alerts)} 个预警")
    
    for alert in alerts:
        print(f"   - {alert.alert_type}: {alert.message} (级别: {alert.risk_level.value})")
    
    # 获取最近预警
    recent_alerts = risk_mgr.get_recent_alerts(max_count=10)
    print(f"\n✅ 最近预警数量: {len(recent_alerts)}")
    
    print("\n✅ 风险预警测试通过!")


def main():
    """主函数"""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "风险监控模块测试" + " " * 40 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        test_basic_operations()
        test_price_updates_and_stops()
        test_closing_positions()
        test_portfolio_metrics()
        test_risk_alerts()
        
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 20 + "✅ 所有测试通过!" + " " * 40 + "║")
        print("╚" + "=" * 78 + "╝")
        
        print("\n📝 风险监控模块功能:")
        print("   - 持仓管理: 开仓、平仓、更新价格")
        print("   - 止损止盈: 自动监控和预警")
        print("   - 组合指标: VaR、最大回撤、集中度、杠杆率")
        print("   - 风险预警: 自动检测组合风险")
        print("   - 净值跟踪: 记录组合净值历史")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
