from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from llm_client import get_llm_response
from aggregator import aggregate_validation
from vector_db import add_context_to_db, search_context
from perplexity import compute_perplexity_metrics
import math

app = FastAPI(title="LLM Response Validation API", version="1.0")

class PerplexityValidation(BaseModel):
    valid: bool
    reason: str
    log_normalized_perplexity_ratio: float

class QueryRequest(BaseModel):
    query: str
    conversation_history: Optional[List[str]] = []
    expected_sentiment: str = "neutral"
    report_content: Optional[str] = None
    original_text: Optional[str] = None

class ValidationResponse(BaseModel):
    llm_response: str
    validator_scores: dict
    average_score: float
    overall_valid: bool
    perplexity_validation: PerplexityValidation

@app.post("/validate", response_model=ValidationResponse)
def validate_query(request: QueryRequest):
    if not request.conversation_history:
        vector_context = search_context(request.query, top_k=3)
    else:
        vector_context = ""
    
    llm_response = get_llm_response(
        request.query,
        report_content=request.report_content,
        original_text=request.original_text
    )
    if not llm_response:
        raise HTTPException(status_code=500, detail="LLM failed to generate a response.")
    
    if request.report_content:
        context_for_perplexity = request.report_content
    elif request.original_text:
        context_for_perplexity = request.original_text
    elif request.conversation_history:
        context_for_perplexity = " ".join(request.conversation_history[-5:])
    else:
        context_for_perplexity = vector_context

    perplexity_metrics = compute_perplexity_metrics(llm_response, context_for_perplexity)
    log_norm_ratio = perplexity_metrics["log_normalized_perplexity_ratio"]
    
    if log_norm_ratio < 0.5:
        ppl_valid = False
        ppl_reason = "Response perplexity is too low—this may indicate an overly generic or repetitive response."
    elif log_norm_ratio > 1.5:
        ppl_valid = False
        ppl_reason = "Response perplexity is too high—this suggests the response is off-topic or incoherent."
    else:
        ppl_valid = True
        ppl_reason = "Response perplexity is within acceptable limits."
    
    scores = aggregate_validation(
        response=llm_response,
        conversation_history=request.conversation_history,
        expected_sentiment=request.expected_sentiment,
        report_content=request.report_content,
        original_text=request.original_text,
        vector_context=vector_context
    )
    
    overall_valid = scores["overall_valid"] and ppl_valid
    
    return ValidationResponse(
        llm_response=llm_response,
        validator_scores=scores,
        average_score=scores["average"],
        overall_valid=overall_valid,
        perplexity_validation=PerplexityValidation(
            valid=ppl_valid,
            reason=ppl_reason,
            log_normalized_perplexity_ratio=log_norm_ratio
        )
    )
