"""Generate baseline model responses with different framing prompts."""
import argparse
import json
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from project.src.prompts import make_system_prompt
from project.src.prompts_fewshot import make_fewshot_prompt, make_simple_prompt
from project.src.model_utils import load_model_and_tokenizer, generate_responses

def load_prompts_from_jsonl(jsonl_path):
    """Load prompts from JSONL file."""
    prompts = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                prompts.append(data['question'])
    return prompts

def main():
    parser = argparse.ArgumentParser(description="Generate baseline responses")
    parser.add_argument("--frame", type=str, default="NEUTRAL", 
                       choices=["NEUTRAL", "PRO", "CON"],
                       help="Framing type")
    parser.add_argument("--num-samples", type=int, default=20,
                       help="Number of prompts to use")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSONL file path")
    parser.add_argument("--model-name", type=str, default="google/flan-t5-base",
                       help="Model name or path")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for inference")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                       help="Max tokens to generate")
    parser.add_argument("--prompts-file", type=str, default=None,
                       help="Path to prompts JSONL file (optional)")
    parser.add_argument("--use-fewshot", action="store_true",
                       help="Use few-shot examples in prompts")
    parser.add_argument("--temperature", type=float, default=None,
                       help="Sampling temperature (enables sampling if set)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get system prompt
    system_prompt = make_system_prompt(args.frame)
    print(f"Frame: {args.frame}")
    print(f"System prompt: {system_prompt}")
    
    # Load model
    print(f"\nLoading model: {args.model_name}")
    model, tokenizer, device = load_model_and_tokenizer(
        args.model_name,
        args.device
    )
    print(f"Device: {device}")
    
    # Load prompts
    if args.prompts_file and Path(args.prompts_file).exists():
        print(f"\nLoading prompts from {args.prompts_file}")
        prompts = load_prompts_from_jsonl(args.prompts_file)
        prompts = prompts[:args.num_samples]
        print(f"Loaded {len(prompts)} prompts from file")
    else:
        # Use default prompts
        DEFAULT_PROMPTS = [
            "Should cities invest more in public transportation?",
            "What are the effects of remote work on productivity?",
            "Is renewable energy economically viable?",
            "Should schools adopt digital textbooks?",
            "What are the impacts of social media on mental health?",
            "Should governments regulate artificial intelligence?",
            "Is nuclear energy a good alternative to fossil fuels?",
            "Should minimum wage be increased?",
            "What are the benefits and drawbacks of globalization?",
            "Should plastic bags be banned?",
            "Is universal basic income a practical policy?",
            "Should college education be free?",
            "What are the effects of automation on employment?",
            "Should genetically modified foods be labeled?",
            "Is electric vehicle adoption feasible?",
            "Should data privacy laws be stricter?",
            "What are the impacts of tourism on local communities?",
            "Should standardized testing be eliminated?",
            "Is cryptocurrency a reliable investment?",
            "Should fast food advertising be restricted?"
        ]
        prompts = DEFAULT_PROMPTS[:args.num_samples]
    
    # Prepare prompts with system frame (with or without few-shot)
    if args.use_fewshot:
        print("Using few-shot examples")
        full_prompts = [make_fewshot_prompt(args.frame, p, system_prompt) for p in prompts]
    else:
        full_prompts = [make_simple_prompt(args.frame, p, system_prompt) for p in prompts]
    
    print(f"\nGenerating {len(prompts)} responses...")
    if args.temperature:
        print(f"Using temperature: {args.temperature}")
    responses = generate_responses(
        model,
        tokenizer,
        full_prompts,
        device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature
    )
    
    # Save results
    print(f"\nSaving to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for idx, (prompt, response) in enumerate(zip(prompts, responses)):
            record = {
                "id": idx,
                "topic": prompt,
                "frame": args.frame,
                "prompt": full_prompts[idx],
                "response": response,
                "meta": {
                    "model": args.model_name,
                    "system_prompt": system_prompt
                }
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✓ Done! Generated {len(responses)} responses")

if __name__ == "__main__":
    main()
