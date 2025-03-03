import json


class InstrumentsService:
    def __init__(self, service):
        service.name = 'InstrumentsService'
        self.manager = service

        self.URL = self.manager.get_url()

    def bonds(self, data=None):
        path = '/Bonds'
        if data is None:
            data = ''

        return self.manager.session.post(
            url=self.URL + path,
            data=data
        ).json()


    def find_instrument(self, query: str, instrumentKind='INSTRUMENT_TYPE_SHARE', apiTradeAvailableFlag=True):
        path = '/FindInstrument'

        return self.manager.session.post(
            url=self.URL + path,
            data=json.dumps({
                'query': query,
                'instrumentKind': instrumentKind,
                'apiTradeAvailableFlag': apiTradeAvailableFlag,
            })
        ).json()


