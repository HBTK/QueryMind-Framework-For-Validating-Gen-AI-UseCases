import re
from typing import List, Dict, Set, Optional

def simple_sentence_split(text: str) -> list:
    """
    Simple and robust sentence tokenization without NLTK dependency.
    """
    # Split on sentence endings while preserving the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Remove empty sentences and strip whitespace
    return [s.strip() for s in sentences if s.strip()]

def simple_word_tokenize(text: str) -> list:
    """
    Simple word tokenization without NLTK dependency.
    """
    # Remove punctuation except apostrophes within words
    text = re.sub(r'[^\w\s\']|\'\s|\s\'|\'+$|^\'+', ' ', text)
    # Split on whitespace and filter out empty strings
    return [word for word in text.split() if word]

def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Compute the Jaccard similarity between two texts.
    """
    try:
        tokens1 = set(simple_word_tokenize(text1.lower()))
        tokens2 = set(simple_word_tokenize(text2.lower()))
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        return len(intersection) / len(union)
    except Exception as e:
        print(f"Error in jaccard_similarity: {str(e)}")
        return 0.0

def check_length(content: str, min_words: int = 300, max_words: int = 2000) -> float:
    """
    Validates content length using simple word tokenization.
    """
    try:
        words = simple_word_tokenize(content)
        count = len(words)
        if min_words <= count <= max_words:
            return 1.0
        return 0.0
    except Exception as e:
        print(f"Error in check_length: {str(e)}")
        return 0.0

def check_structure(content: str, required_sections: Optional[List[str]] = None) -> float:
    """
    Validates presence of required sections using regex.
    """
    try:
        if required_sections is None:
            required_sections = ["title", "introduction", "methodology", "results", "conclusion", "references"]
        
        content_lower = content.lower()
        found = 0
        for section in required_sections:
            pattern = r'\b' + re.escape(section.lower()) + r'\b'
            if re.search(pattern, content_lower):
                found += 1
        return found / len(required_sections)
    except Exception as e:
        print(f"Error in check_structure: {str(e)}")
        return 0.0

def check_formatting(content: str) -> float:
    """
    Validates basic formatting using custom sentence splitting.
    """
    try:
        sentences = simple_sentence_split(content)
        if not sentences:
            return 0.0
        
        correct = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Check if sentence starts with capital letter and ends with proper punctuation
            if sentence[0].isupper() and sentence[-1] in ".!?":
                correct += 1
        return correct / len(sentences)
    except Exception as e:
        print(f"Error in check_formatting: {str(e)}")
        return 0.0

def check_redundancy(content: str) -> float:
    """
    Checks for redundancy using custom sentence normalization.
    """
    try:
        sentences = simple_sentence_split(content)
        if not sentences:
            return 0.0
        
        # Normalize sentences: lowercase, remove punctuation, and extra whitespace
        normalized = [re.sub(r'\W+', ' ', s).strip().lower() for s in sentences]
        unique_sentences = set(normalized)
        return len(unique_sentences) / len(sentences)
    except Exception as e:
        print(f"Error in check_redundancy: {str(e)}")
        return 0.0

def check_reference_format(content: str) -> float:
    """
    Validates reference format using regex patterns.
    """
    try:
        content_lower = content.lower()
        if "references" not in content_lower:
            return 1.0
        
        # Look for URLs or citation patterns
        urls = re.findall(r'https?://[^\s]+', content)
        citations = re.findall(r'\[\d+\]', content)
        return 1.0 if (urls or citations) else 0.0
    except Exception as e:
        print(f"Error in check_reference_format: {str(e)}")
        return 0.0

def check_context_similarity(content: str, context: str, threshold: float = 0.5) -> float:
    """
    Computes context similarity using Jaccard similarity.
    """
    try:
        return jaccard_similarity(content, context)
    except Exception as e:
        print(f"Error in check_context_similarity: {str(e)}")
        return 0.0

def validate_content_generation(content: str, context: str = "", required_sections: Optional[List[str]] = None) -> Dict:
    """
    Aggregates all content validation checks with error handling.
    """
    try:
        scores = {
            "length": check_length(content),
            "structure": check_structure(content, required_sections),
            "formatting": check_formatting(content),
            "redundancy": check_redundancy(content),
            "reference": check_reference_format(content),
            "context_similarity": check_context_similarity(content, context) if context else 1.0
        }
        
        # Calculate average score
        avg_score = sum(scores.values()) / len(scores)
        scores["average"] = round(avg_score, 3)
        scores["overall_valid"] = avg_score >= 0.7
        
        return scores
    except Exception as e:
        print(f"Error in validate_content_generation: {str(e)}")
        return {
            "length": 0.0,
            "structure": 0.0,
            "formatting": 0.0,
            "redundancy": 0.0,
            "reference": 0.0,
            "context_similarity": 0.0,
            "average": 0.0,
            "overall_valid": False
        }