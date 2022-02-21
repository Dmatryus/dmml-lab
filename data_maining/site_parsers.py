from lxml import html
import urllib.request
import ssl
import pandas as pd
import json
from pathlib import Path
from datetime import datetime


class AvitoParser:
    AVITO_URL = 'https://www.avito.ru'

    def __init__(self):
        with open(Path(__file__).parent / 'params.json', encoding='utf-8') as file:
            self.params = json.load(file)

    def _generate_search_url_for(self, city: str, category: str, request_: str, price_from: int = None,
                                 price_to: int = None, page: int = 1):
        url = '/'.join([self.AVITO_URL, city, category])
        request_ = request_.replace(' ', '+')
        params = f'q={request_}'
        if price_from:
            params = f'{params}&pmin={price_from}'
        if price_to:
            params = f'{params}&pmax={price_to}'
        return f'{url}?{params}&p={page}'

    @staticmethod
    def _html_to_tree(link: str):
        try:
            # make ssl certificate (for launch on windows)
            gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)

            with urllib.request.urlopen(link, context=gcontext) as response:
                # write html code to variable
                return html.fromstring(response.read())

        except Exception as err:
            print(link)
            print('Error in html_to_tree')
            print(err.args)

    @staticmethod
    def _get_items_from_tree(tree) -> pd.DataFrame:
        titles = tree.xpath("//div[@data-marker='catalog-serp']//a[@data-marker='item-title']")
        prices = tree.xpath(
            "//div[@data-marker='catalog-serp']//span[@data-marker='item-price']/meta[@itemprop='price']")
        times = tree.xpath("//div[@data-marker='catalog-serp']//div[@data-marker='item-date']")
        items = zip(titles, prices, times)
        items = [i for i in items]
        json_items = []
        for title, price, time in items:
            json_items.append({
                'title': title.get('title'),
                'url': title.get('href'),
                'price': price.get('content'),
                'time': time.text,
                'noticed': datetime.now()
            })
        return pd.DataFrame(json_items)

    def search(self, city: str, category: str, request_: str, price_from: int = None, price_to: int = None):
        page = 0
        city = self.params['city_mapping'][city]
        category = self.params['category_mapping'][category]
        result = pd.DataFrame()
        next_page = True
        while next_page:
            page += 1
            tree = self._html_to_tree(
                self._generate_search_url_for(city, category, request_, price_from, price_to, page))
            if tree is None:
                break
            result = result.append(self._get_items_from_tree(tree))
            next_page = len(
                tree.xpath("//span[@data-marker='pagination-button/next' and not(contains(@class, 'readonly'))]")) > 0
        return result
