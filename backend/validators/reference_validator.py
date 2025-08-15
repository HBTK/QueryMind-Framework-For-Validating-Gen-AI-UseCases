"""
reference_validator.py

This module provides functions to validate external references (URLs) in a text response.
It uses a regex to extract URLs and then sends HTTP HEAD requests to check their existence.
The main functions are:
  - extract_urls: returns a list of URLs found in the input text.
  - reference_validator: returns a normalized score (0 to 1) indicating the proportion of valid URLs.
  - validate_references_details: returns a dictionary mapping each URL to a boolean indicating validity.

Dependencies:
  - requests
  - re (built-in)
"""

import re
import requests

def extract_urls(text: str) -> list:
    """
    Extract all URLs from the given text using a regex pattern.
    
    Parameters:
        text (str): The text from which to extract URLs.
    
    Returns:
        list: A list of URL strings.
    """
    # This regex matches HTTP and HTTPS URLs.
    url_pattern = re.compile(r'https?://[^\s]+')
    return url_pattern.findall(text)

def reference_validator(response: str, timeout: float = 3.0) -> float:
    """
    Validates external references (URLs) in the response by checking their existence using HTTP HEAD requests.
    
    For each URL found in the text, a HEAD request is sent. A URL is considered valid if the HTTP
    response status code is less than 400.
    
    Parameters:
        response (str): The LLM-generated text containing possible URLs.
        timeout (float): Timeout in seconds for each HTTP request (default is 3.0).
    
    Returns:
        float: A score between 0 and 1 representing the proportion of URLs that are valid.
               If no URLs are found, returns 1.0.
    
    Example:
        response = "For details, visit https://www.google.com and https://www.nonexistentwebsite123.com"
        score = reference_validator(response)
    """
    urls = extract_urls(response)
    if not urls:
        return 1.0  # If no URLs, we assume nothing to invalidate.
    valid_count = 0
    for url in urls:
        try:
            # Use a HEAD request for speed.
            r = requests.head(url, timeout=timeout)
            if r.status_code < 400:
                valid_count += 1
        except requests.RequestException:
            # Any exception (e.g., timeout, connection error) counts as invalid.
            continue
    return valid_count / len(urls)

def validate_references_details(response: str, timeout: float = 3.0) -> dict:
    """
    Provides a detailed mapping for each URL found in the response,
    indicating whether it is valid (accessible) or not.
    
    Parameters:
        response (str): The LLM-generated text containing URLs.
        timeout (float): Timeout in seconds for each HTTP request.
    
    Returns:
        dict: A dictionary where keys are URLs and values are booleans (True if the URL is valid).
    
    Example:
        details = validate_references_details("Visit https://www.google.com and https://fakeurl.example")
    """
    urls = extract_urls(response)
    details = {}
    for url in urls:
        valid = False
        try:
            r = requests.head(url, timeout=timeout)
            if r.status_code < 400:
                valid = True
        except requests.RequestException:
            valid = False
        details[url] = valid
    return details

# Example dataset reference (if needed for testing separately):
# You can create a CSV file (e.g., "reference_dataset.csv") with the following content:
#
# url,expected_valid
# https://www.google.com,True
# https://www.wikipedia.org,True
# https://www.nonexistentwebsite1234567890.com,False
#
# Then, using pandas you could load it as follows:
#
# import pandas as pd
# df = pd.read_csv("reference_dataset.csv")
# for idx, row in df.iterrows():
#     print(row['url'], reference_validator(row['url']))
#
# Note: For our validator, the response is assumed to be a text containing URLs.
