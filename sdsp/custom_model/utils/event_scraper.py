from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import pandas as pd
from transformers import pipeline, AutoModel
from datetime import datetime
import requests
import tqdm
import pytz
import h5py
import hashlib 
import numpy as np
import json
from scraper import Config, EthereumNewsScraper

ticker="ETHUSD"
cache_file=f"{ticker}_news_data.hdf5".replace('/', '_')

classification_categories = {
    'Main': ['product launch', 'earnings report', 'merger', 'acquisition', 'CEO change'],
    'Market Movement': ['upward market movement', 'downward market movement'],
    'Regulatory Changes': ['new regulation', 'regulatory change', 'compliance update', 'government policy', 'legislative amendment'],
    'Legal Issues': ['lawsuit', 'legal dispute', 'court ruling', 'regulatory fine', 'legal challenge'],
    'Innovation': ['patent filing', 'research breakthrough', 'technology development', 'R&D update', 'innovation launch'],
    'Management Changes': ['executive appointment', 'management reshuffle', 'new CEO', 'leadership change', 'board restructuring'],
    'Economic Indicators': ['economic outlook', 'market forecast', 'industry report', 'financial indicator', 'economic trend'],
    'Industry Trends': ['market trend', 'industry development', 'sector growth', 'emerging technology', 'market shift'],
    'ESG': ['sustainability initiative', 'social responsibility', 'corporate governance', 'ESG goals', 'environmental impact']
}
# model_sentiment = AutoModel.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

class EventDataScraper:
    def __init__(self, ticker):
        self.ticker = ticker
        # self.sentiment_analysis = pipeline("sentiment-analysis", model='togethercomputer/RedPajama-INCITE-7B-Base')
        # self.sentiment_analysis = pipeline(task="sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", device_map="auto")
                # self.sentiment_analysis = pipeline("sentiment-analysis", model='togethercomputer/RedPajama-INCITE-7B-Base')
        self.sentiment_analysis = pipeline(model="federicopascual/finetuning-sentiment-model-3000-samples")
        # self.event_classifier = pipeline("zero-shot-classification", model="llama2")
        self.event_classifier = pipeline(task="zero-shot-classification",model="facebook/bart-large-mnli")
        self.cache_file = cache_file
        self.hdf5_init()

    def hdf5_init(self):
        with h5py.File(self.cache_file, 'a') as hdf5_file:
            if 'news' not in hdf5_file:
                hdf5_file.create_group('news')
            if 'class_categories' not in hdf5_file:
                class_cat_group = hdf5_file.create_group("class_categories")
            else:
                class_cat_group = hdf5_file['class_categories']

            # Storing each category as a separate dataset
            for category, items in classification_categories.items():
                # Convert list to string
                items_str = ','.join(items)
                # Create or update the dataset
                if category in class_cat_group:
                    del class_cat_group[category]
                class_cat_group.create_dataset(category, data="TEST")
                class_cat_group[category][...] = items_str 


    def open_hdf5_file(self):
        self.hdf5_file = h5py.File(self.cache_file, 'a')

    def close_hdf5_file(self):
        self.hdf5_file.close()
        
    def get_article_hash(self, url):
        # Create a unique hash for each article URL
        return hashlib.md5(url.encode()).hexdigest()

    def convert_time(self, time_str):
        now = datetime.now(pytz.timezone('US/Eastern'))
        if 'Today' in time_str:
            time_str = time_str.replace('Today', now.strftime('%b-%d-%y'))
        elif len(time_str.split(' ')) == 1:
            time_str = now.strftime('%b-%d-%y ') + time_str
        return datetime.strptime(time_str, '%b-%d-%y %I:%M%p')
    
    def get_news_data(self, soup):
        news_data = []

        for news_item in soup.find_all('tr', class_='cursor-pointer has-label'):
            headline_data, headline = news_item.find('a', class_='tab-link-news'), None
            if headline_data:
                headline = headline_data.get_text().strip()
            title_tag = news_item.find('a', class_='tab-link-news')
            source_tag = news_item.find('span')
             # Find the time element
            time_tag = news_item.find('td', align="right")
            
            if title_tag and source_tag:
                title = title_tag.get_text().strip()
                url = title_tag['href']
                time = time_tag.text.strip()  # Extract the text from the time tag
                source = source_tag.get_text().strip('()')
                news_data.append({'title': title, 'url': url, 'source': source, 'headline': headline, 'time': time, 'conv_time': self.convert_time(time)})

        # Sorting the news data by time
        news_data.sort(key=lambda x: x['conv_time'])
        return news_data
    
    def convert_time_to_datetime(self, time_str):
        if 'Today' in time_str:
            time_str = time_str.replace('Today', datetime.now().strftime('%b-%d-%y'))
        return datetime.strptime(time_str, '%b-%d-%y %I:%M%p')
    
    def fetch_article_content(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            # Here, you would need to identify the HTML element that contains the article's text
            # This is an example and might need adjustment based on the actual HTML structure
            article_content = soup.find('div', class_='article-content').get_text()
            return article_content
        except Exception as e:
            return "Error fetching article content: " + str(e)
        
    def find_main_content_div(self, url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        max_length = 0
        main_content = None

        # Iterate over all div elements and find the one with the most text
        for div in soup.find_all('div'):
            text_length = len(div.get_text(strip=True))
            if text_length > max_length:
                max_length = text_length
                main_content = div

        return main_content

    def get_news_data_and_sentiment(self, news_parsed):
        classified_events = {}
        # Organize classification categories into a dictionary for more efficient processing
        classified_events["categories"] = classification_categories
        self.open_hdf5_file()
        news_hdf5_file = self.hdf5_file['news']
        for news in tqdm.tqdm(news_parsed):

            # hdf5 parsing 
            article_hash = hashlib.md5(news['url'].encode()).hexdigest()
            if article_hash in news_hdf5_file:
                news_item = news_hdf5_file[article_hash][...].item()
                news_json_dict = json.loads(news_item)
                keys = [*news_json_dict.keys()]
                sentiment, classification_result, content = keys[1], keys[2], keys[3]
            else:
                article_content = self.fetch_article_content(news['url'])
                sentiment = self.sentiment_analysis(article_content) if article_content else {'label': 'LABEL_2', 'score': 0}
                soup = self.find_main_content_div(news['url'])
                if soup:
                    content = soup.get_text(strip=True)
                    # Initialize classification result
                    classification_result = {}
                    # Classify content under each category
                    for category, classes in tqdm.tqdm(classification_categories.items()):
                        if content:
                            classification = self.event_classifier(content, candidate_labels=classes)
                            classification_result[category] = {'labels': classification['labels'], 'scores': classification['scores']}
                        else:
                            classification_result[category] = {'labels': [], 'scores': []}
                else:
                    print("No main content div found")

                serialized_data = json.dumps({'url': news['url'],'sentiment': sentiment,'classification_result': classification_result,'content': content})
                news_hdf5_file.create_dataset(article_hash,data=np.string_(serialized_data))
                
            result = {'sentiment': sentiment, 'classification': classification_result, 'content': content}
            classified_events[news['url']] = result
        
        self.close_hdf5_file()
        return classified_events
    
    def generate_dataframe(self, news_parsed, classified_events, columns):
        # Create a DataFrame to store combined data
        data = []
        for news in news_parsed:
            import pdb; pdb.set_trace()
            news_url = news['url']
            sentiment_data = classified_events[news_url]['sentiment']
            classification_data = classified_events[news_url]['classification']
            content_data = classified_events[news_url]['content']
            # Combine data for each news article
            combined_data = {
                'title': news['title'],
                'url': news['url'],
                'source': news['source'],
                'time': news['time'],
                'conv_time': news['conv_time'],
                'headline': news['headline'],
                'sentiment': sentiment_data,
                'classification': classification_data,
                'content': content_data
            }
            data.append(combined_data)

        return pd.DataFrame(data, columns=columns)

    def fetch_and_classify_news(self):
        url = f'https://finviz.com/quote.ashx?t={self.ticker}&p=d'
        # import pdb; pdb.set_trace()
        request = Request(url=url, headers={'user-agent': 'news_scraper'})
        response = urlopen(request)
        html = BeautifulSoup(response, features='html.parser')
        news_parsed = self.get_news_data(html)
        classified_events = self.get_news_data_and_sentiment(news_parsed)
        dataframe = self.generate_dataframe(news_parsed, classified_events, columns=['title', 'url', 'source', 'time', 'conv_time', 'headline', 'sentiment', 'classification', 'content'])
        return dataframe
    
    def classify_articles(self, news_parsed, ticker):
        classified_events = self.get_news_data_and_sentiment(news_parsed)
        dataframe = self.generate_dataframe(news_parsed, classified_events, columns=['title', 'url', 'source', 'time', 'conv_time', 'headline', 'sentiment', 'classification', 'content'])
        return dataframe


# Example usage
# scraper = EventDataScraper(ticker)
# df_events = scraper.fetch_and_classify_news()
ticker = "ETHUSD"
ethscraper = EthereumNewsScraper(api_key=Config.API_KEY)
news_parsed = ethscraper.load_data(Config.JSON_SAVE_PATH)
scraper = EventDataScraper(ticker)
df_events = scraper.classify_articles(news_parsed, ticker)

df_events.to_csv(f'{ticker}_events_data.csv', index=False)
df_events.to_hdf(f'{ticker}_events_data.h5', key='df', mode='w')