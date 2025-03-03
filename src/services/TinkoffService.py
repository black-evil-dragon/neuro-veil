from session import Session


class TinkoffService:

    is_sandbox = False

    SANDBOX_URL = "https://sandbox-invest-public-api.tinkoff.ru/rest"
    PRODUCTION_URL = "https://invest-public-api.tinkoff.ru/rest"

    URL = SANDBOX_URL if is_sandbox else PRODUCTION_URL
    TOKEN = ''

    session = None

    name = ''
    path = 'tinkoff.public.invest.api.contract.v1'


    def __init__(self, token='', is_sandbox=False):
        self.is_sandbox = is_sandbox
        self.TOKEN = token


        self.URL = self.SANDBOX_URL if self.is_sandbox else self.PRODUCTION_URL

        self.session = Session(
            url=self.get_url(),
            token=self.TOKEN
        )


    def get_url(self):
        return f'{self.URL}/{self.path}.{self.name}'


