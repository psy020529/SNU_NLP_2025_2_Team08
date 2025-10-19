# Placeholder for scoring functions.
PRO_LEX = {"benefit","advantage","efficient","cost-effective","improve","increase"}
CON_LEX = {"risk","drawback","inefficient","costly","harm","decrease"}

def framing_score(text: str) -> float:
    t = text.lower()
    pro = sum(w in t for w in PRO_LEX)
    con = sum(w in t for w in CON_LEX)
    L = max(1, len(text.split()))
    return (pro - con)/L

def balance_score(text: str) -> float:
    t = text.lower()
    pro = sum(w in t for w in PRO_LEX)
    con = sum(w in t for w in CON_LEX)
    return 1.0 - min(1.0, abs(pro-con)/max(1, pro+con))
