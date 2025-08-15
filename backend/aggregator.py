"""
Aggregator Module for LLM Response Validation

This module combines scores from various rule-based validators:
  1. Common validators:
     - Sentiment (sentiment_validator)
     - Ethical considerations (ethical_validator)
     - Bias & Fairness (bias_validator)
     - Privacy & Security (privacy_validator)
     - Context-based similarity (context_similarity_validator)
     - Reference validations (reference_validator)
     
  2. Specialized validators (for specific use cases):
     - Report content validation (report_content_validator)
     - Summarization validation (summarization_validator)
     
Each validator returns a score between 0 and 1. The aggregator collects all available scores,
computes an overall average (skipping any that aren’t applicable), and flags the response as valid
if the overall average is above a threshold.
"""

import numpy as np
from validators import (
    sentiment_validator,
    ethical_validator,
    bias_validator,
    privacy_validator,
    context_similarity_validator,
    reference_validator,
    report_content_validator,
    summarization_validator,
)

def convert_to_python(value):
    """
    Converts numpy numeric or boolean types to native Python types.
    """
    if value is None:
        return value
    if isinstance(value, (np.bool_, np.number)):
        return value.item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value

def get_dynamic_context(conversation_history: list, original_text: str, report_content: str) -> str:
    """
    Determines the reference context based on use case:
      - For summarization: return original_text.
      - For report generation: return report_content.
      - For conversational AI: if conversation_history is provided, return the last 5 messages.
      - Otherwise, return an empty string.
    """
    if original_text:
        return original_text  # Summarization
    elif report_content:
        return report_content  # Report Generation
    elif conversation_history:
        return " ".join(conversation_history[-5:])
    return ""

def aggregate_validation(response: str,
                           conversation_history: list = None,
                           expected_sentiment: str = "neutral",
                           report_content: str = None,
                           original_text: str = None,
                           vector_context: str = "",  # New parameter
                           overall_threshold: float = 0.7) -> dict:
    """
    Aggregates validation scores from multiple rule-based validators.
    
    Parameters:
      - response (str): LLM-generated response.
      - conversation_history (list, optional): Chat history (for conversational use cases).
      - expected_sentiment (str): Expected sentiment ("neutral", "positive", or "negative").
      - report_content (str, optional): Reference text for report generation validation.
      - original_text (str, optional): Reference text for summarization validation.
      - vector_context (str, optional): Context retrieved from a vector DB, used if other context is empty.
      - overall_threshold (float): Minimum average score required for overall validity.
    
    Returns:
      dict: Contains individual validator scores, an "average" score, and an "overall_valid" flag.
    """
    # Determine dynamic context based on use case
    context = get_dynamic_context(conversation_history, original_text, report_content)
    # If no context was found and vector_context is provided, use that.
    if not original_text and not report_content and vector_context.strip():
        context = vector_context

    scores = {}
    # Common validators
    scores["sentiment"] = convert_to_python(sentiment_validator.is_sentiment_valid(response, expected_sentiment))
    scores["ethical"] = convert_to_python(ethical_validator.ethical_score(response))
    
    bias_result = bias_validator.bias_score(response)
    scores["bias"] = convert_to_python(bias_result.get("score", 0.0)) if isinstance(bias_result, dict) else convert_to_python(bias_result)
        
    scores["privacy"] = convert_to_python(privacy_validator.validate_privacy(response))
    
    if context.strip():
        cs_score = context_similarity_validator.context_similarity_score(response, context)
        scores["context_similarity"] = convert_to_python(cs_score)
    else:
        scores["context_similarity"] = None

    scores["reference"] = convert_to_python(reference_validator.reference_validator(response))
    
    # Specialized validators (if provided)
    if report_content:
        content_result = report_content_validator.validate_content_generation(report_content, context)
        scores["report_content"] = convert_to_python(content_result.get("average", 0.0))
    else:
        scores["report_content"] = None

    if original_text:
        summ_result = summarization_validator.summarize_validation(original_text, response)
        scores["summarization"] = convert_to_python(summ_result.get("average_score", 0.0))
    else:
        scores["summarization"] = None

    # Compute overall average from scores that are numbers (skip None)
    valid_scores = [float(convert_to_python(score)) for score in scores.values() if isinstance(convert_to_python(score), (int, float))]
    overall_avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    scores["average"] = round(overall_avg, 3)
    scores["overall_valid"] = overall_avg >= overall_threshold

    return scores
