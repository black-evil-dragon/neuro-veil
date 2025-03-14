from models.model import Model


class TradingBot:
    def __init__(self, model, initial_balance=10000, position_size=0.1, take_profit=0.02, stop_loss=0.01):
        self.model = model
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position_size = position_size
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.position = None  # None, 'long', or 'short'
        self.entry_price = None
        self.trades = []
        self.current_step = 0

    def reset(self):
        self.balance = self.initial_balance
        self.position = None
        self.entry_price = None
        self.trades = []
        self.current_step = 0

    def execute_trade(self, action, price):
        if action == 'buy' and self.position is None:
            self.position = 'long'
            self.entry_price = price
            self.trades.append({'step': self.current_step, 'action': 'buy', 'price': price})
        elif action == 'sell' and self.position == 'long':
            profit = (price - self.entry_price) / self.entry_price
            self.balance *= (1 + profit * self.position_size)
            self.position = None
            self.entry_price = None
            self.trades.append({'step': self.current_step, 'action': 'sell', 'price': price, 'profit': profit})
        elif action == 'hold':
            pass

    def check_stop_loss_take_profit(self, price):
        if self.position == 'long':
            profit = (price - self.entry_price) / self.entry_price
            if profit >= self.take_profit or profit <= -self.stop_loss:
                self.execute_trade('sell', price)

    def run(self, data, features):
        self.reset()
        for i in range(len(data) - self.model.look_back):
            self.current_step = i
            current_data = data[i:i + self.model.look_back]
            print(features)
            prediction = self.model.predict(current_data, features)
            action = self.decide_action(prediction)
            current_price = current_data[-1]['close']
            self.execute_trade(action, current_price)
            self.check_stop_loss_take_profit(current_price)
        return self.trades, self.balance

    def decide_action(self, prediction):
        # Пример простой стратегии: покупать, если предсказание выше текущей цены, и продавать, если ниже
        current_price = prediction[-1]
        predicted_price = prediction[-1]  # Используем последнее предсказание
        if predicted_price > current_price:
            return 'buy'
        elif predicted_price < current_price:
            return 'sell'
        else:
            return 'hold'
