import os
import re
import nltk
from afinn import Afinn

# Download essential NLTK data
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize AFINN and VADER
afinn = Afinn()
vader = SentimentIntensityAnalyzer()

def simple_tokenize(text: str) -> list:
    """
    A simple, robust tokenization function that lowercases, replaces common contractions,
    and splits text into words.
    """
    text = text.lower()
    # Replace common contractions
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"'d", " would", text)
    # Extract words (alphanumeric)
    words = re.findall(r'\b\w+\b', text)
    return words

def afinn_sentiment_score(response: str) -> float:
    """
    Computes sentiment score using the Afinn lexicon.
    The raw score is normalized to a value between 0 (very negative) and 1 (very positive).
    """
    if not response or not response.strip():
        return 0.5  # Neutral for empty response
    try:
        words = simple_tokenize(response)
        if not words:
            return 0.5
        raw_score = sum(afinn.score(word) for word in words)
        max_possible_score = len(words) * 5  # assuming max score of 5 per word
        min_possible_score = len(words) * -5
        if max_possible_score == min_possible_score:
            return 0.5
        normalized_score = (raw_score - min_possible_score) / (max_possible_score - min_possible_score)
        return round(max(0.0, min(normalized_score, 1.0)), 3)
    except Exception as e:
        print(f"Error in afinn_sentiment_score: {str(e)}")
        return 0.5

def vader_sentiment_score(response: str) -> float:
    """
    Computes sentiment score using NLTK's VADER.
    VADER returns a compound score between -1 and 1, which we normalize to [0,1].
    """
    if not response or not response.strip():
        return 0.5
    try:
        scores = vader.polarity_scores(response)
        compound = scores.get('compound', 0)
        normalized = (compound + 1) / 2  # Normalize from [-1,1] to [0,1]
        return round(max(0.0, min(normalized, 1.0)), 3)
    except Exception as e:
        print(f"Error in vader_sentiment_score: {str(e)}")
        return 0.5

def combined_sentiment_score(response: str) -> float:
    """
    Computes the combined sentiment score as the average of the Afinn and VADER scores.
    """
    score_afinn = afinn_sentiment_score(response)
    score_vader = vader_sentiment_score(response)
    combined = (score_afinn + score_vader) / 2
    return round(combined, 3)

def is_sentiment_valid(response: str, expected: str = "neutral", threshold: float = 0.5) -> float:
    """
    Validates the sentiment of the response against the expected sentiment.
    
    For "neutral", the best score is near 0.5;
    for "positive", higher scores are better;
    for "negative", lower scores are better.
    
    Returns a validity score between 0 and 1.
    """
    score = combined_sentiment_score(response)
    
    if expected == "neutral":
        # Closer to 0.5 is best; scale difference to [0,1]
        validity = 1 - abs(score - 0.5) * 2  
    elif expected == "positive":
        validity = score
    elif expected == "negative":
        validity = 1 - score
    else:
        raise ValueError("Expected sentiment must be 'neutral', 'positive', or 'negative'.")
    
    return round(max(0.0, min(validity, 1.0)), 3)
