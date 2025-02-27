from datetime import timedelta
from http.client import responses

from config import SANDBOX_TOKEN

import json

from services.TinkoffService import TinkoffService

from services.InstrumentsService import InstrumentsService
from services.MarketDataService import MarketDataService
from utils.time import now, prepare_date

TService = TinkoffService(token=SANDBOX_TOKEN, is_sandbox=True)

instrumentsService = InstrumentsService(TService)
marketDataService = MarketDataService(TService)


# response = instrumentsService.bonds()


response = instrumentsService.find_instrument('RU000A107UL4')
tbank_instrument = response.get('instruments')[0]



tbank_candles = marketDataService.get_candles(
    from_date=prepare_date(now() - timedelta(days=1)),
    instrumentId=tbank_instrument.get('figi'),
    limit=2,
)

print(tbank_candles)
# [print(tbank_candles[c]) for c in tbank_candles]



