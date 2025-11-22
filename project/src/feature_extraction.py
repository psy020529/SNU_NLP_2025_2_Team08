"""Enhanced feature extraction for better detection."""
import re
import numpy as np
from collections import Counter

# Positive and negative lexicons (expanded)
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'beneficial',
    'advantage', 'benefit', 'opportunity', 'success', 'improvement', 'progress',
    'innovation', 'effective', 'efficient', 'valuable', 'promising', 'positive',
    'helpful', 'useful', 'important', 'significant', 'thriving', 'flourishing',
    'vibrant', 'exciting', 'revolutionary', 'breakthrough', 'cutting-edge'
}

NEGATIVE_WORDS = {
    'bad', 'poor', 'terrible', 'awful', 'horrible', 'problematic', 'harmful',
    'disadvantage', 'drawback', 'risk', 'failure', 'decline', 'concern',
    'challenge', 'difficult', 'ineffective', 'inefficient', 'costly', 'expensive',
    'dangerous', 'threat', 'loss', 'damage', 'struggle', 'crisis', 'plagued',
    'burden', 'questionable', 'skeptical', 'cautionary', 'serious', 'underestimate'
}

HEDGE_WORDS = {
    'might', 'may', 'could', 'possibly', 'perhaps', 'arguably', 'somewhat',
    'relatively', 'fairly', 'moderately', 'tend', 'suggest', 'indicate'
}

CERTAINTY_WORDS = {
    'definitely', 'certainly', 'obviously', 'clearly', 'undoubtedly', 'absolutely',
    'always', 'never', 'must', 'will', 'guaranteed', 'proven', 'fact'
}

def extract_enhanced_features(text):
    """Extract comprehensive features for classification."""
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    if not words:
        return {
            'length': 0,
            'avg_word_length': 0,
            'positive_ratio': 0,
            'negative_ratio': 0,
            'sentiment_diff': 0,
            'hedge_ratio': 0,
            'certainty_ratio': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'first_person_count': 0,
            'modal_verb_count': 0,
            'comparative_count': 0,
            'superlative_count': 0
        }
    
    # Basic metrics
    length = len(words)
    avg_word_length = np.mean([len(w) for w in words])
    
    # Sentiment lexicons
    positive_count = sum(1 for w in words if w in POSITIVE_WORDS)
    negative_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    positive_ratio = positive_count / length
    negative_ratio = negative_count / length
    sentiment_diff = positive_ratio - negative_ratio
    
    # Certainty and hedging
    hedge_count = sum(1 for w in words if w in HEDGE_WORDS)
    certainty_count = sum(1 for w in words if w in CERTAINTY_WORDS)
    hedge_ratio = hedge_count / length
    certainty_ratio = certainty_count / length
    
    # Punctuation and style
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # First person (advocacy indicator)
    first_person = sum(1 for w in words if w in ['i', 'we', 'our', 'us', 'my'])
    first_person_ratio = first_person / length
    
    # Modal verbs (uncertainty/possibility)
    modal_verbs = sum(1 for w in words if w in ['should', 'would', 'could', 'might', 'may', 'can'])
    modal_ratio = modal_verbs / length
    
    # Comparatives and superlatives (often used in advocacy)
    comparatives = len(re.findall(r'\b\w+er\b', text_lower))
    superlatives = len(re.findall(r'\b\w+est\b', text_lower))
    comparative_ratio = comparatives / length
    superlative_ratio = superlatives / length
    
    return {
        'length': length,
        'avg_word_length': avg_word_length,
        'positive_ratio': positive_ratio,
        'negative_ratio': negative_ratio,
        'sentiment_diff': sentiment_diff,
        'hedge_ratio': hedge_ratio,
        'certainty_ratio': certainty_ratio,
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'first_person_ratio': first_person_ratio,
        'modal_ratio': modal_ratio,
        'comparative_ratio': comparative_ratio,
        'superlative_ratio': superlative_ratio
    }
