"""
Ethical Validator (Improved, Combined Techniques)

This module validates the ethical content of a response by checking for banned/offensive phrases.
It uses several traditional techniques:
  1. Fuzzy Matching: Uses difflib.SequenceMatcher over n-grams to detect phrases similar to banned ones.
  2. Keyword Frequency Ratio: Calculates the ratio of banned tokens present in the response.
  3. TF-IDF Cosine Similarity: Computes the similarity between the response and a corpus
     of banned phrases (inverted so that lower similarity means better ethical score).

The final ethical score is computed as the average of these three scores.
A score of 1.0 means the response is completely ethically acceptable, while lower values indicate more issues.

Preferred External Dataset URL:
    https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en

Dependencies:
    - Standard Python libraries: os, re, difflib, requests, io, csv
    - Optional: pandas (if you want to load from CSV)
    - scikit-learn (for TF-IDF and cosine similarity)
    - To install pandas and scikit-learn: pip install pandas requests scikit-learn
"""

import os
import re
import difflib
import requests
import io
import csv

try:
    import pandas as pd
except ImportError:
    pd = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Default list of banned/offensive phrases (fallback)
DEFAULT_BIASED_PHRASES = [
    "all women are", 
    "all men are",
    "women are naturally", 
    "men are naturally",
    "you throw like a girl",
    "all black people are",
    "black people are inherently",
    "all asians are",
    "asians are naturally",
    "all latinos are",
    "all muslims are",
    "muslims are terrorists",
    "all christians are",
    "christians are judgmental",
    "old people are useless",
    "elderly people are weak",
    "all seniors are slow",
    "all immigrants are criminals",
    "all americans are ignorant",
    "all europeans are arrogant",
    "all africans are",
    "poor people are lazy",
    "rich people are snobbish",
    "all gay people are",
    "all homosexuals are",
    "all lesbians are",
    "all queer people are",
    "all conservatives are",
    "all liberals are",
    "all feminists are",
    "all [group] are"
]

# External dataset URL (plain text file with one term per line)
EXTERNAL_BIAS_URL = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en"

# -----------------------------
# Data Loading Functions
# -----------------------------
def load_external_bias_data(url: str) -> list:
    """
    Downloads bias terms from the provided URL.
    Expects one term per line.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        terms = [line.strip().lower() for line in response.text.splitlines() if line.strip()]
        return terms
    except Exception as e:
        print(f"Error loading external bias dataset from {url}: {e}")
        return []

def load_bias_csv_from_string(csv_string: str) -> list:
    """
    Loads bias terms from an embedded CSV string with header 'term'.
    """
    reader = csv.DictReader(io.StringIO(csv_string))
    terms = [row["term"].strip().lower() for row in reader if row["term"].strip()]
    return terms

def get_bias_phrases(custom_phrases: list = None, csv_path: str = None, external_url: str = EXTERNAL_BIAS_URL) -> list:
    """
    Returns a combined list of biased phrases from multiple sources.
    Priority:
      1. custom_phrases (if provided)
      2. phrases loaded from CSV (if csv_path provided)
      3. phrases from the external dataset
      4. embedded default phrases
    """
    phrases = set()
    
    if custom_phrases is not None and isinstance(custom_phrases, list) and custom_phrases:
        phrases.update([p.strip().lower() for p in custom_phrases if p.strip()])
    
    if csv_path is not None:
        try:
            df = pd.read_csv(csv_path)
            if "phrase" in df.columns:
                phrases.update(df["phrase"].dropna().astype(str).str.strip().str.lower().tolist())
        except Exception as e:
            print(f"Error loading CSV: {e}")
    
    external_phrases = load_external_bias_data(external_url)
    if external_phrases:
        phrases.update([p.strip().lower() for p in external_phrases if p.strip()])
    
    embedded_phrases = load_bias_csv_from_string("term\n" + "\n".join(DEFAULT_BIASED_PHRASES))
    if embedded_phrases:
        phrases.update([p.strip().lower() for p in embedded_phrases if p.strip()])
    
    return sorted(list(phrases))

# -----------------------------
# Technique 1: Fuzzy Matching
# -----------------------------
def tokenize(text: str) -> list:
    """
    Tokenizes input text into lowercase alphanumeric tokens.
    """
    return re.findall(r'\w+', text.lower())

def ngrams(tokens: list, n: int) -> list:
    """
    Generates overlapping n-grams from a list of tokens.
    """
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def fuzzy_match(phrase: str, text: str, threshold: float = 0.85) -> bool:
    """
    Uses fuzzy matching to check if the phrase is present in text.
    Returns True if any n-gram's similarity is above the threshold.
    """
    phrase = phrase.lower().strip()
    phrase_tokens = tokenize(phrase)
    n = len(phrase_tokens)
    if n == 0:
        return False
    text_tokens = tokenize(text)
    for gram in ngrams(text_tokens, n):
        ratio = difflib.SequenceMatcher(None, phrase, gram).ratio()
        if ratio >= threshold:
            return True
    return False

def fuzzy_ethics_score(response: str, bias_phrases: list) -> float:
    """
    Computes a fuzzy ethics score using fuzzy matching.
    Returns 1/(1+count) where count is the number of biased phrases matched.
    """
    count = 0
    for phrase in bias_phrases:
        if fuzzy_match(phrase, response):
            count += 1
    return 1.0 / (1.0 + count)

# -----------------------------
# Technique 2: Keyword Frequency Ratio
# -----------------------------
def keyword_frequency_ratio(response: str, bias_phrases: list) -> float:
    """
    Computes the ratio of banned tokens in the response.
    Returns 1.0 if no banned tokens are found, else 1/(1+count).
    """
    tokens = tokenize(response)
    if not tokens:
        return 1.0
    count = sum(1 for token in tokens if token in set(bias_phrases))
    return 1.0 if count == 0 else 1.0 / (1.0 + count)

# -----------------------------
# Technique 3: TF-IDF Cosine Similarity
# -----------------------------
def tfidf_cosine_similarity(response: str, bias_phrases: list) -> float:
    """
    Computes the TF-IDF cosine similarity between the response and a document
    constructed by concatenating all bias phrases.
    Returns an inverted score: lower similarity yields a higher ethical score.
    """
    try:
        # Concatenate all bias phrases into a single document
        bias_doc = " ".join(bias_phrases)
        documents = [response, bias_doc]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        # Invert similarity so that lower similarity (i.e., less biased) gets higher score
        return max(0.0, min(1.0, 1 - sim))
    except Exception as e:
        print(f"Error in TF-IDF cosine similarity: {e}")
        return 1.0

# -----------------------------
# Final Ethical Score Aggregation
# -----------------------------
def ethical_score(response: str, custom_phrases: list = None, csv_path: str = None, external_url: str = EXTERNAL_BIAS_URL) -> float:
    """
    Evaluates the ethical content of the response using a combination of techniques.
    Returns a final score between 0 and 1:
      - 1 indicates no biased or offensive content detected,
      - Lower values indicate more bias.
      
    Techniques used:
      1. Fuzzy Matching: Computes 1/(1+count) of matches.
      2. Keyword Frequency Ratio: Computes 1/(1+count) of exact banned token matches.
      3. TF-IDF Cosine Similarity (inverted): Computes 1 - similarity between the response and banned phrases corpus.
      
    The final ethical score is the average of these three metrics.
    """
    bias_phrases = get_bias_phrases(custom_phrases, csv_path, external_url)
    
    score_fuzzy = fuzzy_ethics_score(response, bias_phrases)
    score_keyword = keyword_frequency_ratio(response, bias_phrases)
    score_tfidf = tfidf_cosine_similarity(response, bias_phrases)
    
    final_score = (score_fuzzy + score_keyword + score_tfidf) / 3
    return round(final_score, 3)

def is_ethically_valid(response: str, threshold: float = 0.5, **kwargs) -> bool:
    """
    Returns True if the ethical score is above the threshold.
    Additional keyword arguments are passed to ethical_score.
    """
    return ethical_score(response, **kwargs) >= threshold

# Example usage (for testing):
if __name__ == "__main__":
    test_response = "All women are naturally weak and all immigrants are criminals."
    score = ethical_score(test_response)
    print("Final Ethical Score:", score)
