from watchdog.strategies import get_builtin_strategies
strategies = get_builtin_strategies()
for strategy in strategies:
    print(f"{strategy.name} ({strategy.id}): {strategy.description}")