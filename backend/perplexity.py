import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"

def load_model_and_tokenizer(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def calculate_perplexity(text: str) -> float:
    """
    Tokenizes the text, computes model logits, and returns perplexity.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs.input_ids[:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss)
    return perplexity.item()

def compute_perplexity_metrics(response: str, context: str) -> dict:
    """
    Computes perplexity for the LLM response and the provided context,
    then calculates:
      - Perplexity of response
      - Perplexity of context
      - The ratio (response / context)
      - The log-normalized perplexity ratio.
    Returns these values in a dictionary.
    """
    ppl_response = calculate_perplexity(response)
    ppl_context = calculate_perplexity(context)
    ppl_ratio = ppl_response / ppl_context if ppl_context > 0 else float('inf')
    epsilon = 1e-10
    log_norm_ratio = math.log(max(ppl_response, epsilon)) / math.log(max(ppl_context, epsilon))
    return {
        "perplexity_response": round(ppl_response, 3),
        "perplexity_context": round(ppl_context, 3),
        "perplexity_ratio": round(ppl_ratio, 3),
        "log_normalized_perplexity_ratio": round(log_norm_ratio, 4)
    }
