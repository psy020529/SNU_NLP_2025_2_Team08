# Placeholder for prompt framing templates.
NEUTRAL_SYS = "You are a balanced assistant. Present pros and cons fairly."
PRO_SYS     = "Advocate supportive framing. Start with benefits; mention downsides briefly."
CON_SYS     = "Advocate critical framing. Start with drawbacks; mention benefits briefly."

def make_system_prompt(frame: str) -> str:
    return {"NEUTRAL": NEUTRAL_SYS, "PRO": PRO_SYS, "CON": CON_SYS}[frame]
