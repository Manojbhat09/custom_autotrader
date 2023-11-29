from bs4 import BeautifulSoup
import requests

def scrape_crypto_news(url):
    # Sending a request to the URL
    response = requests.get(url)
    if response.status_code != 200:
        return "Failed to retrieve the webpage"

    # Parsing the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')
    import pdb; pdb.set_trace()
    # Finding news article links - update the selector as per the actual page structure
    articles = soup.find_all('div', class_='media-body')  # This selector should be adjusted based on the actual HTML
    news_links = [article.find('a')['href'] for article in articles if article.find('a')]

    return news_links

# URL of the page to scrape
url = "https://seekingalpha.com/market-news/crypto?page=13"

# Scraping and printing the news links
news_links = scrape_crypto_news(url)
for link in news_links:
    print(link)
