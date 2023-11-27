import requests
from bs4 import BeautifulSoup

def find_main_content_div(url):
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

url = 'https://finance.yahoo.com/video/jeff-bezos-amazon-shares-vinfast-220122547.html'
main_div = find_main_content_div(url)

# Output the text content of the div with the most text
if main_div:
    print(main_div.get_text(strip=True))
else:
    print("No main content div found")
import pdb; pdb.set_trace()
