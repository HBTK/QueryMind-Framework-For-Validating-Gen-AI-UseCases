# File: web_scraper.py

import requests
from bs4 import BeautifulSoup
from googlesearch import search

# List of allowed domains that are known for high-quality content.
ALLOWED_DOMAINS = [
    "wikipedia.org", 
    "arxiv.org", 
    "medium.com", 
    "researchgate.net", 
    "blogspot.com"
]

def get_web_context(query: str, num_results: int = 1) -> str:
    """
    Searches for the query on Google restricted to allowed domains (e.g., Wikipedia, arXiv, Medium, ResearchGate, Blogspot).
    Returns the main textual content from the first result that meets a minimal length requirement.
    
    Parameters:
      - query: The search query.
      - num_results: Number of search results to retrieve per domain (default is 1).
    
    Returns:
      A string containing the text content (from paragraph tags) of the chosen webpage,
      or an empty string if no valid result is found.
    """
    for domain in ALLOWED_DOMAINS:
        # Modify the query to restrict results to the current allowed domain.
        modified_query = f"{query} site:{domain}"
        try:
            results = list(search(modified_query, num_results=num_results, lang="en"))
            if results:
                url = results[0]
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    # Extract text from all <p> tags.
                    paragraphs = soup.find_all("p")
                    content = " ".join(p.get_text() for p in paragraphs)
                    # Return content only if it meets a minimal length requirement.
                    if content and len(content) > 100:
                        return content
        except Exception as e:
            print(f"Error searching on {domain}: {e}")
    return ""
