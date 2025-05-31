import wikipediaapi
import os
import requests
from bs4 import BeautifulSoup
from apis.utils import wiki_split_html

def search_wiki(query: str, max_results: int = 5) -> list[str]:
    """
    Search Wikipedia for a given query and return a list of titles of the top results.
    """
    
    # convert query to url friendly format using requests
    formatted_query = requests.utils.quote(query)
    endpoint = f"https://en.wikipedia.org/w/index.php?limit={max_results}&offset=0&search={formatted_query}&title=Special:Search"
    result = requests.get(endpoint)
    
    # check if the request was successful
    if result.status_code != 200:
        raise Exception(f"Failed to fetch results from Wikipedia: {result.status_code}")
    
    # parse the response to extract titles
    titles = []
    soup = BeautifulSoup(result.text, 'html.parser')
    for item in soup.select('.mw-search-result-heading a'):
        titles.append(item.get_text())
    
    return titles

wiki_wiki = wikipediaapi.Wikipedia(
    user_agent = f'RateCast ({os.getenv("EMAIL_ADDRESS", "")})',
    language = 'en',
    extract_format=wikipediaapi.ExtractFormat.HTML
)

def get_wiki_summary(title: str) -> str:
    """
    Get a summary of a Wikipedia page by its title.
    """
    
    page = wiki_wiki.page(title)
    
    if not page.exists():
        return f"Page '{title}' does not exist on Wikipedia."
    
    return page.summary

def get_wiki_full_text(title: str) -> str:
    """
    Fetches full rendered HTML (including tables) from a Wikipedia article using the MediaWiki API.
    """
    headers = {
        'User-Agent': f'RateCast ({os.getenv("EMAIL_ADDRESS", "")})'
    }

    # Step 1: Get page ID from title
    base_url = 'https://en.wikipedia.org/w/api.php'
    page_params = {
        'action': 'parse',
        'page': title,
        'format': 'json',
        'prop': 'text'
    }

    response = requests.get(base_url, params=page_params, headers=headers)
    data = response.json()

    if 'error' in data:
        return f"Error fetching page: {data['error']}"

    html_content = data['parse']['text']['*']
    return html_content

def get_wiki_links(title: str) -> list[str]:
    """
    Get all links from a Wikipedia page by its title.
    """
    
    page = wiki_wiki.page(title)
    
    if not page.exists():
        raise ValueError(f"Page '{title}' does not exist on Wikipedia.")
    
    return list(page.links.keys())

def get_wiki_full_text_batched(title: str) -> list[str]:
    try:
        page_text = get_wiki_full_text(title)
        batched_text = [batch.get('extracted_text') for batch in wiki_split_html(page_text)]
        return batched_text
    except ValueError:
        return []

