
import re

PROMPT_SUFFIX_RE = re.compile(r"\s*question\s*:\s*is\s+the\s+passage\s+above\s+toxic\??", re.IGNORECASE)

def strip_prompt_suffix(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return PROMPT_SUFFIX_RE.sub("", text)

def tokenize_words(text: str) -> list[str]:
    base = strip_prompt_suffix(text)
    return re.findall(r"[a-z0-9]+", base.lower())

def clean_text(text: str) -> str:
    return " ".join(tokenize_words(text))

def normalize_text(text: str) -> str:
    return "".join(tokenize_words(text))

def make_key(text: str, num_words: int = 8) -> str:
    words = tokenize_words(text)
    return " ".join(words[:num_words])