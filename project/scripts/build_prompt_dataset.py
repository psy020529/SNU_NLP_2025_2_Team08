"""Build a larger prompt dataset from public sources."""
import json
import random
from pathlib import Path
from datasets import load_dataset

def extract_neutral_topics_from_mm_framing(num_samples=100):
    """Extract neutral topics from mm-framing dataset."""
    print("Loading mm-framing dataset...")
    ds = load_dataset('copenlu/mm-framing', split='full', streaming=True)
    
    topics = set()
    neutral_titles = []
    
    for i, sample in enumerate(ds):
        if i >= 5000:  # Limit to first 5000 to avoid long processing
            break
            
        # Filter for center/neutral political leaning and specific topics
        if sample.get('political_leaning') in ['center', 'left_lean', 'right_lean']:
            title = sample.get('title', '').strip()
            topic = sample.get('gpt-topic', '').strip()
            
            if title and topic and len(title) > 20 and len(title) < 150:
                # Convert news headlines to questions
                if not title.endswith('?'):
                    # Transform to question format
                    question = convert_headline_to_question(title)
                    if question and question not in topics:
                        topics.add(question)
                        neutral_titles.append({
                            'question': question,
                            'topic': topic,
                            'source': 'mm-framing',
                            'original_leaning': sample.get('political_leaning')
                        })
        
        if len(neutral_titles) >= num_samples:
            break
    
    return neutral_titles

def convert_headline_to_question(headline):
    """Convert news headline to a neutral question."""
    # Remove leading/trailing quotes
    headline = headline.strip('"').strip("'").strip()
    
    # Skip if too short or already a question
    if len(headline) < 15 or headline.endswith('?'):
        return None
    
    # Simple conversion patterns
    conversions = [
        # Statement -> Question patterns
        ("is ", "Is "),
        ("are ", "Are "),
        ("will ", "Will "),
        ("should ", "Should "),
        ("can ", "Can "),
        ("has ", "Has "),
        ("have ", "Have "),
    ]
    
    # Try to detect statements that can be converted
    lower_headline = headline.lower()
    
    # If headline starts with action verb, try to convert
    if any(lower_headline.startswith(pattern) for pattern, _ in conversions):
        # Already looks like a question structure
        if not headline.endswith('?'):
            return headline + '?'
        return headline
    
    # For other headlines, create "What are the implications of" format
    return f"What are the implications of {headline.lower()}?"

def extract_topics_from_social_bias(num_samples=50):
    """Extract discussion topics from social bias frames."""
    print("Loading social_bias_frames dataset...")
    try:
        ds = load_dataset('allenai/social_bias_frames', split='train', streaming=True)
        
        topics = []
        seen = set()
        
        for i, sample in enumerate(ds):
            if i >= 1000:
                break
            
            # Extract target category and convert to neutral question
            target_category = sample.get('targetCategory', '').strip()
            if target_category and target_category not in seen:
                seen.add(target_category)
                # Create neutral questions about these topics
                question = f"What are the social implications of policies affecting {target_category}?"
                topics.append({
                    'question': question,
                    'topic': target_category,
                    'source': 'social_bias_frames',
                    'original_leaning': 'neutral'
                })
            
            if len(topics) >= num_samples:
                break
        
        return topics
    except Exception as e:
        print(f"Warning: Could not load social_bias_frames: {e}")
        return []

def generate_synthetic_questions(num_samples=50):
    """Generate synthetic neutral questions on common topics."""
    templates = [
        "What are the effects of {} on society?",
        "How does {} impact daily life?",
        "What are the benefits and drawbacks of {}?",
        "Should governments regulate {}?",
        "Is {} economically viable in the long term?",
        "What are the environmental implications of {}?",
        "How can {} be made more accessible?",
        "What role does {} play in modern society?",
    ]
    
    topics = [
        "artificial intelligence", "renewable energy", "remote work",
        "online education", "electric vehicles", "social media",
        "automation", "cryptocurrency", "universal healthcare",
        "data privacy", "genetically modified foods", "space exploration",
        "nuclear energy", "universal basic income", "telemedicine",
        "smart cities", "5G technology", "gene editing",
        "autonomous vehicles", "carbon taxes"
    ]
    
    questions = []
    for _ in range(num_samples):
        template = random.choice(templates)
        topic = random.choice(topics)
        question = template.format(topic)
        questions.append({
            'question': question,
            'topic': topic,
            'source': 'synthetic',
            'original_leaning': 'neutral'
        })
    
    return questions

def main():
    output_dir = Path("nlp-proj/data/prompts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Building prompt dataset from multiple sources...")
    
    # Collect from multiple sources
    all_prompts = []
    
    # 1. MM-Framing dataset
    print("\n1. Extracting from mm-framing...")
    mm_prompts = extract_neutral_topics_from_mm_framing(num_samples=100)
    all_prompts.extend(mm_prompts)
    print(f"   Collected {len(mm_prompts)} prompts from mm-framing")
    
    # 2. Social Bias Frames
    print("\n2. Extracting from social_bias_frames...")
    sbf_prompts = extract_topics_from_social_bias(num_samples=50)
    all_prompts.extend(sbf_prompts)
    print(f"   Collected {len(sbf_prompts)} prompts from social_bias_frames")
    
    # 3. Synthetic generation
    print("\n3. Generating synthetic prompts...")
    synthetic_prompts = generate_synthetic_questions(num_samples=50)
    all_prompts.extend(synthetic_prompts)
    print(f"   Generated {len(synthetic_prompts)} synthetic prompts")
    
    # Deduplicate
    unique_prompts = []
    seen_questions = set()
    for prompt in all_prompts:
        q = prompt['question'].lower().strip()
        if q not in seen_questions:
            seen_questions.add(q)
            unique_prompts.append(prompt)
    
    print(f"\nTotal unique prompts: {len(unique_prompts)}")
    
    # Save as JSONL
    output_file = output_dir / "all_prompts.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(unique_prompts):
            prompt['id'] = i
            f.write(json.dumps(prompt, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Saved {len(unique_prompts)} prompts to {output_file}")
    
    # Create a smaller sample for quick testing
    sample_file = output_dir / "sample_prompts.jsonl"
    with open(sample_file, 'w', encoding='utf-8') as f:
        for prompt in unique_prompts[:30]:
            f.write(json.dumps(prompt, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved 30 sample prompts to {sample_file}")

if __name__ == "__main__":
    main()
