import json
from utils.numbers import parse_quotation

class DataProcessor:
    @staticmethod
    def process_indicators(indicators):
        processed_data = []
        for indicator in indicators['technicalIndicators']:
            processed_data.append({
                'time': indicator['timestamp'],
                'value': parse_quotation(indicator['signal']),      
            })
        return processed_data
    
    
    @staticmethod
    def process_candles(candles):
        processed_data = []
        for candle in candles['candles']:
            processed_data.append({
                'time': candle['time'],
                # 'price': parse_quotation(candle['close']),
                'open': parse_quotation(candle['open']),
                'high': parse_quotation(candle['high']),
                'low': parse_quotation(candle['low']),
                'close': parse_quotation(candle['close']),
                'volume': int(candle['volume']),
                # 'isComplete': candle['isComplete'],
                # 'candleSourceType': candle['candleSourceType'],
                
            })
        return processed_data
    
    @staticmethod
    def save_to_json(data, filename):
        """Сохраняет данные в JSON файл."""
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data saved to {filename}")

    @staticmethod
    def load_from_json(filename):
        """Загружает данные из JSON файла."""
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        print(f"Data loaded from {filename}")
        return data