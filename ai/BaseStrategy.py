from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    def __init__(self, initial_cash=100_000):
        self.initial_cash = initial_cash
        self.reset()

    def reset(self):
        self.cash = self.initial_cash
        self.shares = 0
        self.in_trade = False
        self.buy_markers = []
        self.sell_markers = []
        self.trades = 0

    @abstractmethod
    def run(self, stock_data, model, scaler, *args, **kwargs):
        pass

    def result(self, stock_data):
        final_value = self.cash + self.shares * stock_data["Close"].iloc[-1]
        profit_ratio = (final_value - self.initial_cash) / self.initial_cash
        return {
            "name": self.__class__.__name__,
            "buy": self.buy_markers,
            "sell": self.sell_markers,
            "final_value": final_value,
            "num_trades": self.trades,
            "profit_ratio": profit_ratio,
        }
