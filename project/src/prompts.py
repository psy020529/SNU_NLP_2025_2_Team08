# Placeholder for prompt framing templates.
TEMPLATES = {
    "NEUTRAL" : "You are a balanced assistant. Present pros and cons fairly.",
    "PRO" : "Advocate supportive framing. Start with benefits; mention downsides briefly.",
    "CON" : "Advocate critical framing. Start with drawbacks; mention benefits briefly."
}

def make_system_prompt(frame: str) -> str:
    return TEMPLATES.get(frame.upper(), TEMPLATES["NEUTRAL"])
