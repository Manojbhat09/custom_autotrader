from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time 

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dateutil.relativedelta import relativedelta
import datetime
import json, requests
from tqdm import tqdm

class Config:
    HEADLESS_OPTION = True
    SEE_MORE_COUNT = 5
    CSS_SELECTOR_BUTTON = "button[class*='BaseButton_base__aMbeB']"
    CSS_SELECTOR_NEWS_CONTAINER = "div[data-direction='ltr']"
    SLEEP_INTERVAL = 10
    DRIVER_WAIT_TIME = 10
    API_ENDPOINT = 'https://api.search.brave.com/res/v1/web/search'
    API_HEADER_ACCEPT = 'application/json'
    API_HEADER_ACCEPT_ENCODING = 'gzip'
    API_PARAM_QUERY = 'q'
    API_PARAM_FRESHNESS = 'freshness'
    DATE_PARSE_FUNCTION = 'parse_date'
    REQUEST_TIMEOUT = 10
    USER_AGENT = 'Mozilla/5.0 (compatible; ScraperBot/0.1)'
    NEWS_DATE_FORMAT = '%Y-%m-%d'
    API_KEY = 'BSA-nASPATEcJuL8FJvBrM_2IWmvQ-y'
    QUERY = 'Ethereum USA news ETH'
    CMCURL = 'https://coinmarketcap.com/currencies/ethereum/#News'
    JSON_SAVE_PATH = 'ethereum_news_2.json'
    DURATION = 100

class EthereumNewsScraper:
    def __init__(self, api_key):
        self.api_key = api_key
        options = Options()
        options.headless = Config.HEADLESS_OPTION
        self.driver = webdriver.Chrome(options=options)

    def fetch_news_cmc(self, url):
        if 'coinmarketcap' not in url:
            print("Link must be from coinmarketcap")
            return

        self.driver.get(url)
        time.sleep(5)  # Wait for JavaScript to load

        try:
            count = 0
            while count < Config.SEE_MORE_COUNT:
                count += 1
                see_more_button = WebDriverWait(self.driver, Config.DRIVER_WAIT_TIME).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, Config.CSS_SELECTOR_BUTTON))
                )
                
                if see_more_button:
                    self.driver.execute_script("arguments[0].scrollIntoView();", see_more_button)
                    self.driver.execute_script("arguments[0].click();", see_more_button)
                    print("Clicked 'See More' button")
                    time.sleep(Config.SLEEP_INTERVAL)  # Wait for new content to load
                else:
                    break
        except Exception as e:
            print("Error or end of page reached:", e)
        
        print("Completed searching for content")
        news_containers = WebDriverWait(self.driver, Config.DRIVER_WAIT_TIME).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, Config.CSS_SELECTOR_NEWS_CONTAINER))
        )
        news_data = self.process_news_containers(news_containers)
        self.driver.quit()
        print("Total new articles:", len(news_data))
        return news_data

    def parse_date(self, date_text):
        current_time = datetime.datetime.now()
        if 'ago' in date_text:
            # Split the text and get the first part (e.g., "an hour", "2 hours")
            number, unit = date_text.split()[:2]
            # Convert the number part to an integer, defaulting to 1 if it's "an"
            number = 1 if number == "an" else int(number)
            # Subtract the appropriate timedelta based on the unit
            if 'hour' in unit:
                return current_time - datetime.timedelta(hours=number)
            # Add more cases for days, weeks, etc. if needed
        # Handle other cases if needed
        return current_time  # Default to current time if no time is given

    
    def process_news_containers(self, news_containers):
        news_data = []
        last_date = None
        for container in news_containers:
            parts = container.text.split('\n')
            if 'ago' in parts[0]:
                last_date = self.parse_date(parts[0])
            date_text = last_date if last_date else datetime.now()

            title = parts[1] if len(parts) >= 2 else 'No Title'
            description = parts[2] if len(parts) >= 3 else 'No Description'
            try:
                url = container.find_element(By.TAG_NAME, 'a').get_attribute('href')
            except Exception as e:
                print(f"Error extracting URL: {e}")
                url = None
            
            news_data.append({
                'date': date_text, # Convert to string for consistent formatting
                'title': title,
                'description': description,
                'url': url,
            })

        return news_data

    def process_news_containers(self, news_containers):
        news_data = []
        for container in news_containers:
            # Split the text by newline to separate the date, title, and description
            parts = container.text.split('\n')
            if len(parts) >= 3:  # Ensure there are at least three parts
                date_text = parts[0]
                title = parts[1]
                description = '\n'.join(parts[2:])  # Join the remaining parts as the description

                # Attempt to extract the URL
                try:
                    url = container.find_element(By.TAG_NAME, 'a').get_attribute('href')
                except Exception as e:
                    print(f"Error extracting URL: {e}")
                    url = None  

                # Add to the list
                news_data.append({
                    'date': date_text,  # You might want to convert this to a datetime object
                    'title': title,
                    'description': description,
                    'url': url,
                })

        return news_data
    
    def fetch_brave_search_eth_news(self,  query, from_date, to_date):
        # This defines a function with parameters for the API key, search query, and date range.
        
        url = 'https://api.search.brave.com/res/v1/web/search'
        # The URL for the Brave Web Search API endpoint.

        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.api_key
        }
        # Headers for the request, including the API key in the 'X-Subscription-Token' field.

        params = {
            'q': query,
            'freshness': f'{from_date}to{to_date}'
        }
        # Parameters for the request. 'q' is the query term, and 'freshness' filters results based on the provided date range.

        response = requests.get(url, headers=headers, params=params)
        # Sending a GET request to the API with the specified URL, headers, and parameters.

        if response.status_code != 200:
            return "Failed to retrieve data from Brave Search API"
        # Checking if the response status code is not 200 (OK). If not, it returns an error message.

        return response.json()
        # If the response is successful, it returns the JSON response from the API.

    def parse_date(self, page_age):
        """
        Convert 'page_age' string to a datetime object.
        This function needs to be adjusted based on the actual format of 'page_age'.
        """
        current_time = datetime.datetime.now()
        if 'ago' in page_age:
            # Example parsing for formats like '2 weeks ago', '1 month ago'
            number, unit = re.match(r'(\d+)\s(\w+)', page_age).groups()
            number = int(number)

            if 'week' in unit:
                return current_time - relativedelta(weeks=number)
            elif 'month' in unit:
                return current_time - relativedelta(months=number)
            elif 'day' in unit:
                return current_time - relativedelta(days=number)
            # Add more conditions if there are other formats like 'year'
        else:
            # If 'page_age' is in a specific date format
            return datetime.datetime.strptime(page_age, '%Y-%m-%dT%H:%M:%S')

    def extract_data_from_link(self, link):
        try:
            # Send a request to the link
            response = requests.get(link)
            response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
            
            # Parse the content with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title, description, and date
            # You'll need to adjust the selectors based on the actual structure of the web page
            title = soup.find('h1', class_='news-title').get_text()
            description = soup.find('div', class_='news-content').get_text()
            date_string = soup.find('time', class_='news-date').get('datetime')  # Assuming the date is in a 'datetime' attribute
            date = datetime.datetime.fromisoformat(date_string)
            
            return title, description, date
        except Exception as e:
            print(f"Error extracting data from {link}: {e}")
            return None, None, None

    def scrape_and_fetch_news(self, url, query, duration=60):
        duration = Config.DURATION
        # Fetch news from coinmarketcap
        news_data = self.fetch_news_cmc(url)

        # Fetch news using Brave Search API
        from_date = (datetime.datetime.now() - datetime.timedelta(days=duration)).strftime('%Y-%m-%d')
        to_date = datetime.datetime.now().strftime('%Y-%m-%d')
        api_news = self.fetch_brave_search_eth_news(query, from_date, to_date)

        # Combine and format the data
        extracted_data = []

        # Add data from Brave Search API results
        for result in api_news['web']['results']:
            parsed_date = self.parse_date(result['page_age'])
            extracted_data.append({
                'title': result['title'],
                'url': result['url'],
                'description': result['description'],
                'date': parsed_date
            })
        # For example, you can sort the combined data by date
        extracted_data.extend(news_data)
        # extracted_data = sorted(extracted_data, key=lambda x: x['date'], reverse=True)
        
        return extracted_data

    def save_data(self, data, filename):
        # Function to format datetime objects for JSON serialization
        def json_serial(obj):
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        # Save data to a JSON file
        with open(filename, 'w') as file:
            json.dump(data, file, default=json_serial)

    def load_data(self, filename):
        # Load data from a JSON file
        with open(filename, 'r') as file:
            return json.load(file)

if __name__ == "__main__":
    api_key =  Config.API_KEY
    scraper = EthereumNewsScraper(api_key=api_key)
    query = Config.QUERY
    url = Config.CMCURL
    extracted_data = scraper.scrape_and_fetch_news(url, query)
    scraper.save_data(extracted_data, Config.JSON_SAVE_PATH)
