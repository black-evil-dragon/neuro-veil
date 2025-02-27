import requests


class Session:
    def __init__(self, url: str, token: str):
        self.url = url
        self.session = requests.Session()
        self.token = token

        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }

        self.session.headers = self.headers



    def get(self, url: str) -> requests.Response:
        return self.session.get(url)


    def post(self, url: str, data='') -> requests.Response:
        return self.session.post(url, data=data, json=data)