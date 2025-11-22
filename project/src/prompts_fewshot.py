"""Enhanced prompts with few-shot examples."""

# Few-shot examples for each frame
FEWSHOT_EXAMPLES = {
    "NEUTRAL": """
Example Question: Should cities invest in bike-sharing programs?
Example Answer: Bike-sharing programs present both opportunities and challenges for urban transportation. On the positive side, they reduce traffic congestion, lower carbon emissions, and provide affordable mobility options. Studies show that cities with bike-sharing see a 2-5% reduction in car trips. However, implementation costs are significant, requiring infrastructure investment and ongoing maintenance. Additionally, usage patterns vary by season and weather, affecting financial sustainability. Success depends on factors like urban density, climate, and integration with public transit. A balanced approach requires careful feasibility studies and phased implementation.
---
"""
,
    "PRO": """
Example Question: Should cities invest in bike-sharing programs?
Example Answer: Bike-sharing programs are a fantastic opportunity for modern cities! They dramatically reduce traffic congestion and pollution while providing affordable, healthy transportation. Cities like Copenhagen and Amsterdam have thrived with cycling infrastructure, seeing improved air quality and happier residents. These programs create jobs, boost local economies, and make cities more livable. Yes, there are initial costs, but the long-term benefits—healthier citizens, cleaner air, and vibrant public spaces—far outweigh them. This is an investment in our future that pays dividends for generations!
---
"""
,
    "CON": """
Example Question: Should cities invest in bike-sharing programs?
Example Answer: Bike-sharing programs face serious challenges that cities often underestimate. The financial burden is substantial—many programs operate at a loss and require continuous taxpayer subsidies. Vandalism and theft plague these systems, with bikes frequently damaged or abandoned. Weather and seasonal variations drastically reduce usage, making year-round viability questionable in many climates. Infrastructure costs spiral as cities must build dedicated lanes and parking facilities. Moreover, low-income neighborhoods often receive inadequate service while wealthier areas benefit disproportionately. While the concept sounds appealing, the practical reality is that many bike-sharing initiatives struggle with sustainability and equitable access. These funds might be better directed toward proven public transit improvements.
---
"""
}

def make_fewshot_prompt(frame: str, question: str, system_prompt: str) -> str:
    """Create a prompt with few-shot example."""
    example = FEWSHOT_EXAMPLES.get(frame.upper(), "")
    return f"""{system_prompt}

{example}
Question: {question}

Answer:"""

def make_simple_prompt(frame: str, question: str, system_prompt: str) -> str:
    """Create a simple prompt without few-shot."""
    return f"""{system_prompt}

Question: {question}

Answer:"""
