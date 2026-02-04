# Task 786 Fix: Translation Function with Arbitrary Prefix Strings

## IFE Summary

The hidden reference solution is **not a real translation function**. It is a toy implementation that:
- Returns `'Spanish - ' + text` for `language='es'`
- Returns `'French - ' + text` for `language='fr'`
- Returns `text` unchanged for `language='en'`
- Returns the literal string `'Unsupported language'` for any other code

This is an arbitrary, undocumented behavior that no agent could infer from the task description ("translate it into other languages... like the ones they use in NYC government communications").

## Evidence

Hidden information code:
```python
def translate_text(text, language):
    return {'es': lambda text: 'Spanish - ' + text,
            'fr': lambda text: 'French - ' + text,
            'en': lambda text: text}.get(language, lambda text: 'Unsupported language')(text)
```

The simulated user cannot explain why "translation" means "prefix with language name" because:
1. This is an arbitrary design choice not derivable from the task
2. The user prompt says to say "I don't know" if info cannot be found

## Fix Strategy

The test cases likely expect exact string matches to the hidden implementation. This is fundamentally a **specification defect**: the task asks for "translation" but tests for "prefixing."

**Recommended fix**: Update the instruction to clarify that this is a **labeling function**, not a translation function, and specify the exact output format expected.

## Fix Applied

- `instruction_override.json`: Clarifies that this is a mock/labeling function with specific output format requirements
