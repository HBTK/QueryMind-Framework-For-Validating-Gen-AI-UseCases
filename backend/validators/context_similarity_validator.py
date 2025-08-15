# File: context_similarity_validator.py

import re
from io import StringIO
import pandas as pd

# For TF-IDF and cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Jaccard Similarity Function
# ----------------------------
def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Compute the Jaccard similarity between two texts.
    Returns a score between 0 and 1, where 1 means identical token sets.
    """
    tokens1 = set(re.findall(r'\w+', text1.lower()))
    tokens2 = set(re.findall(r'\w+', text2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)

# ----------------------------
# TF-IDF Cosine Similarity Function
# ----------------------------
def tfidf_cosine_similarity(text1: str, text2: str) -> float:
    """
    Computes cosine similarity between two texts using TF-IDF vectorization.
    Returns a score between 0 and 1.
    """
    try:
        vectorizer = TfidfVectorizer()
        corpus = [text1, text2]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return cos_sim[0][0]
    except Exception as e:
        print(f"Error computing TF-IDF cosine similarity: {e}")
        return 0.0

# ----------------------------
# Combined Similarity Function
# ----------------------------
def combined_similarity(text1: str, text2: str, weight_jaccard: float = 0.5, weight_tfidf: float = 0.5) -> float:
    """
    Computes a combined similarity score as a weighted average of:
      - Jaccard similarity and 
      - TF-IDF cosine similarity.
    Weights can be adjusted; by default, both are weighted equally.
    """
    jac_score = jaccard_similarity(text1, text2)
    tfidf_score = tfidf_cosine_similarity(text1, text2)
    combined = weight_jaccard * jac_score + weight_tfidf * tfidf_score
    return combined

# ----------------------------
# Context-Based Similarity Validator
# ----------------------------
def context_similarity_score(response: str, context: str) -> float:
    """
    Computes similarity between the generated response and the reference context
    using the combined similarity measure.
    """
    return combined_similarity(response, context)

def validate_context_similarity(response: str, context: str, threshold: float = 0.5) -> bool:
    """
    Validates if the LLM response is contextually relevant.
    Returns True if the combined similarity score is at or above the threshold; otherwise, False.
    """
    score = context_similarity_score(response, context)
    return score >= threshold

# ----------------------------
# Dynamic Context Handling
# ----------------------------
def get_dynamic_context(conversation_history: list, original_text: str, report_content: str) -> str:
    """
    Determines the reference context based on use case:
      - Summarization: Uses original_text.
      - Report Generation: Uses report_content.
      - Conversational AI: Uses the last 5 messages from conversation history.
    If none are provided, returns an empty string.
    """
    if original_text:
        return original_text  # For summarization
    elif report_content:
        return report_content  # For report generation
    elif conversation_history:
        return " ".join(conversation_history[-5:])
    return ""
