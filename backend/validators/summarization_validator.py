"""
summarization_validator.py

This module implements a validator for text summarization use cases.
It computes several scores:
  1. Length Score: Ideal summary length relative to the original text.
  2. Coverage Score: Keyword coverage from the original text.
  3. Redundancy Score: How non-redundant the summary is.
  4. BLEU Score: Measures n-gram overlap between summary and original.
  5. ROUGE Score: Uses ROUGE-L F1 to compare summary with original.
  6. (Optional) BERTScore: Semantic similarity (F1 score) computed using pretrained BERT.

An overall average score is computed as the average of the computed metrics.
Set include_bertscore=False (default) to avoid downloading large model files.

Dependencies:
  - nltk (pip install nltk)
  - rouge-score (pip install rouge-score)
  - bert-score (pip install bert-score) [if include_bertscore is True]
  - Standard libraries: re, string, collections
"""

import re
import string
from collections import Counter

# For BLEU
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# For ROUGE
from rouge_score import rouge_scorer

# For BERTScore (only used if enabled)
try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None

# Download required NLTK data (if not already downloaded)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------- Helper Functions ----------

def simple_tokenize(text: str) -> list:
    """
    A simple robust tokenization function that lowercases, removes punctuation,
    and splits on whitespace.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return words

def simple_sentence_split(text: str) -> list:
    """
    Simple sentence tokenization based on punctuation.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def get_stopwords() -> set:
    """
    Returns a basic set of English stopwords.
    """
    return {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
        'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
        'will', 'with', 'this', 'but', 'they', 'have', 'had', 'what', 'when',
        'where', 'who', 'which', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'can', 'just', 'should', 'now', 'i',
        'you', 'your', 'we', 'my', 'me', 'her', 'his', 'their', 'our', 'us', 'am'
    }

# ---------- Existing Rule-Based Metrics ----------

def summarization_length_score(original: str, summary: str, target_range: tuple = (0.2, 0.5)) -> float:
    """
    Computes a score based on the ratio of summary length to original text length.
    Returns 1.0 if within target_range, lower if too short or too long.
    """
    try:
        original_words = simple_tokenize(original)
        summary_words = simple_tokenize(summary)
        if not original_words:
            return 0.0
        ratio = len(summary_words) / len(original_words)
        low, high = target_range
        if low <= ratio <= high:
            return 1.0
        if ratio < low:
            return ratio / low
        return max(0.0, 1 - (ratio - high) / (1 - high))
    except Exception as e:
        print(f"Error in length score calculation: {str(e)}")
        return 0.0

def extract_keywords(text: str, top_n: int = 10) -> list:
    """
    Extracts top keywords from text using frequency analysis.
    """
    try:
        stop_words = get_stopwords()
        words = simple_tokenize(text)
        words = [w for w in words if w not in stop_words]
        word_freq = Counter(words)
        keywords = [word for word, _ in word_freq.most_common(top_n)]
        return keywords
    except Exception as e:
        print(f"Error in keyword extraction: {str(e)}")
        return []

def coverage_score(original: str, summary: str, top_n: int = 10) -> float:
    """
    Computes a score based on how many of the top keywords in the original text appear in the summary.
    """
    try:
        keywords = extract_keywords(original, top_n)
        if not keywords:
            return 1.0
        summary_words = set(simple_tokenize(summary))
        hits = sum(1 for kw in keywords if kw in summary_words)
        return hits / len(keywords)
    except Exception as e:
        print(f"Error in coverage score calculation: {str(e)}")
        return 0.0

def redundancy_score(summary: str) -> float:
    """
    Evaluates redundancy in the summary by comparing unique sentences vs total sentences.
    """
    try:
        sentences = simple_sentence_split(summary)
        if not sentences:
            return 1.0
        unique_sentences = set(s.lower() for s in sentences)
        return len(unique_sentences) / len(sentences)
    except Exception as e:
        print(f"Error in redundancy score calculation: {str(e)}")
        return 0.0

# ---------- Additional Automatic Metrics ----------

def compute_bleu_score(original: str, summary: str) -> float:
    """
    Computes a BLEU score between the original and summary.
    Uses NLTK's sentence_bleu with smoothing.
    """
    try:
        reference = [simple_tokenize(original)]
        candidate = simple_tokenize(summary)
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu(reference, candidate, smoothing_function=smoothing)
        return min(bleu, 1.0)
    except Exception as e:
        print(f"Error in BLEU score calculation: {str(e)}")
        return 0.0

def compute_rouge_score(original: str, summary: str) -> float:
    """
    Computes ROUGE-L F1 score using the rouge_score package.
    """
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(original, summary)
        rouge_l_f = scores['rougeL'].fmeasure
        return rouge_l_f
    except Exception as e:
        print(f"Error in ROUGE score calculation: {str(e)}")
        return 0.0

def compute_bertscore(original: str, summary: str, include_bertscore: bool = False) -> float:
    """
    Computes BERTScore F1 between original and summary.
    If include_bertscore is False, returns a neutral default value of 0.5.
    """
    if not include_bertscore:
        return 0.5
    if bert_score is None:
        print("bert_score package is not available.")
        return 0.0
    try:
        P, R, F1 = bert_score([summary], [original], lang="en", verbose=False)
        return F1.item()
    except Exception as e:
        print(f"Error in BERTScore calculation: {str(e)}")
        return 0.0

# ---------- Aggregator for Summarization Validation ----------

def summarize_validation(original: str, summary: str, include_bertscore: bool = False) -> dict:
    """
    Aggregates various summarization validation scores:
      - Length score
      - Coverage score
      - Redundancy score
      - BLEU score
      - ROUGE score
      - BERTScore F1 (if include_bertscore is True; otherwise, a neutral value of 0.5)
      
    Returns a dictionary with each individual score and an average score.
    """
    try:
        length = summarization_length_score(original, summary)
        coverage = coverage_score(original, summary)
        redundancy = redundancy_score(summary)
        bleu = compute_bleu_score(original, summary)
        rouge = compute_rouge_score(original, summary)
        bert = compute_bertscore(original, summary, include_bertscore)
        
        # Create a list of scores to average
        scores_list = [length, coverage, redundancy, bleu, rouge, bert]
        avg_score = sum(scores_list) / len(scores_list)
        
        return {
            "length_score": round(length, 3),
            "coverage_score": round(coverage, 3),
            "redundancy_score": round(redundancy, 3),
            "bleu_score": round(bleu, 3),
            "rouge_score": round(rouge, 3),
            "bertscore": round(bert, 3),
            "average_score": round(avg_score, 3)
        }
    except Exception as e:
        print(f"Error in summarization validation: {str(e)}")
        return {
            "length_score": 0.0,
            "coverage_score": 0.0,
            "redundancy_score": 0.0,
            "bleu_score": 0.0,
            "rouge_score": 0.0,
            "bertscore": 0.0,
            "average_score": 0.0
        }
