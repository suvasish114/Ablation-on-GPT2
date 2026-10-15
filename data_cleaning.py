import re

def clean_text(sentence: str) -> str:
    text = re.sub(r"[^a-zA-Z]+", " ", sentence)
    return " ".join(text.lower().split())