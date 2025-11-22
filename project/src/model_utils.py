"""Model loading and inference utilities."""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Optional

def load_model_and_tokenizer(
    model_name: str = "google/flan-t5-base",
    device: Optional[str] = None
):
    """Load model and tokenizer."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_new_tokens: int = 256,
    batch_size: int = 4,
    temperature: Optional[float] = None
) -> List[str]:
    """Generate responses for a list of prompts."""
    responses = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            if temperature is not None:
                # Use sampling with temperature
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9
                )
            else:
                # Use greedy decoding (deterministic)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1
                )
        
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses.extend(batch_responses)
    
    return responses
