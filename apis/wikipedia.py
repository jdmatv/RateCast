import wikipediaapi
import os
import requests
from bs4 import BeautifulSoup

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
    language = 'en'
)

def get_wiki_summary(title: str) -> str:
    """
    Get a summary of a Wikipedia page by its title.
    """
    
    page = wiki_wiki.page(title)
    
    if not page.exists():
        raise ValueError(f"Page '{title}' does not exist on Wikipedia.")
    
    return page.summary

def get_wiki_full_text(title: str) -> str:
    """
    Get the full text of a Wikipedia page by its title.
    """
    
    page = wiki_wiki.page(title)
    
    if not page.exists():
        raise ValueError(f"Page '{title}' does not exist on Wikipedia.")
    
    return page.text

wiki1 = get_wiki_summary(search_wiki("Python programming language", 1)[0])
print(wiki1)


