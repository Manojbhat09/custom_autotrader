from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time 


class EthereumNewsScraper:
    def __init__(self):
        from selenium import webdriver
        options = Options()
        options.headless = True
        self.driver = webdriver.Chrome(options=options)

    def fetch_news_links(self, url):

        if 'coinmarketcap' not in url:
            print("link has to be from coinmarketcap")
            return 
        
        self.driver.get(url)
        # Wait for JavaScript to load
        time.sleep(5)

        html = self.driver.page_source
        self.driver.quit()

        soup = BeautifulSoup(html, features="html.parser")
        import pdb; pdb.set_trace()
        # Select the section containing news descriptions
        news_descriptions = soup.find_all("div", class_="news_description")
        
        links = []
        for description in news_descriptions:
            element = description
            while element:
                href = element.get('href')
                if href:
                    links.append(href)
                    break
                element = element.next_element

        return links

# Example usage
scraper = EthereumNewsScraper()
url = 'https://coinmarketcap.com/currencies/ethereum/#News'
links = scraper.fetch_news_links(url)
print(links)
