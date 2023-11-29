import requests
import datetime

def fetch_brave_search_eth_news(api_key, query, from_date, to_date):
    # This defines a function with parameters for the API key, search query, and date range.
    
    url = 'https://api.search.brave.com/res/v1/web/search'
    # The URL for the Brave Web Search API endpoint.

    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': api_key
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

def parse_date(page_age):
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

# API Key for Brave Search API
api_key = 'BSA-nASPATEcJuL8FJvBrM_2IWmvQ-y'  # Replace with your actual API key

# Define the search query and date range
query = 'Ethereum USA news ETH'
from_date = (datetime.datetime.now() - datetime.timedelta(days=60)).strftime('%Y-%m-%d')
to_date = datetime.datetime.now().strftime('%Y-%m-%d')

# Fetching Ethereum news
eth_news = fetch_brave_search_eth_news(api_key, query, from_date, to_date)

# Extracting and sorting the data
extracted_data = []
for result in eth_news['web']['results']:
    parsed_date = parse_date(result['page_age'])
    extracted_data.append({
        'title': result['title'],
        'url': result['url'],
        'description': result['description'],
        'date': parsed_date
    })

# Sorting the list of dictionaries by date
sorted_data = sorted(extracted_data, key=lambda x: x['date'], reverse=True)
import pdb; pdb.set_trace()
# Now sorted_data contains the sorted news items
print(sorted_data)


