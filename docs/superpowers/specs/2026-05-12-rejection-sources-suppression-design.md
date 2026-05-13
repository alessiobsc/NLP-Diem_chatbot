# Design: Rejection Sources Suppression + Language-Aware Responses

**Date:** 2026-05-12  
**Status:** Approved

## Problem

`chat_stream()` and `chat()` always append source URLs after every response, including scope rejection and knowledge gap responses. Sources after a rejection are semantically wrong — there are no relevant documents to cite.

Additionally, rejection phrases are currently hardcoded in English in `SYSTEM_PROMPT`, causing the LLM to respond in English even when the user writes in Italian.

## Goals

1. Suppress source URLs when the response is a scope rejection or knowledge gap
2. LLM responds in the user's language for all response types (including rejections)

## Approach: Hidden Tag Detection

The LLM prepends a machine-readable tag to rejection responses. The application layer detects, strips, and uses the tag to gate source formatting. The response content itself is free-form and language-independent.

## Changes

### `src/prompts.py`

**Remove** hardcoded exact English phrases from rules 3 and 4 in `SYSTEM_PROMPT`.

**Replace with:**
- Rule 3 (scope rejection): "If the question is completely unrelated to DIEM, start your response with the tag `[FUORI_SCOPE]`, then explain in the user's language."
- Rule 4 (knowledge gap): "If the question is DIEM-related but the context lacks the answer, start your response with the tag `[KNOWLEDGE_GAP]`, then explain in the user's language."

No explicit language instruction added — qwen2.5:7b follows the user's input language natively when not instructed otherwise. Adding an explicit instruction causes the model to default to English (known 7b behavior).

Define in `prompts.py`:
```python
REJECTION_TAGS = ("[FUORI_SCOPE]", "[KNOWLEDGE_GAP]")
```

### `src/brain.py` — `chat_stream()`

Buffer the first `MAX_TAG_LEN + 2` characters before yielding. Once the buffer threshold is reached, check for tag presence, strip if found, then begin yielding. Gate `_format_sources` on `is_rejection`.

```python
MAX_TAG_LEN = max(len(t) for t in REJECTION_TAGS)  # 14

answer = ""
tag_checked = False
is_rejection = False

for chunk in self._chat_model.stream(prompt_value):
    answer += chunk.content
    if not tag_checked and len(answer) >= MAX_TAG_LEN + 2:
        tag_checked = True
        for tag in REJECTION_TAGS:
            if answer.startswith(tag):
                is_rejection = True
                answer = answer[len(tag):].lstrip()
                break
    if tag_checked:
        yield answer

if not is_rejection:
    sources_md = self._format_sources(reranked)
    if sources_md:
        yield answer + sources_md
```

### `src/brain.py` — `chat()`

Post-process the full answer string. Strip tag and gate sources.

```python
import re

REJECTION_TAGS = ...  # imported from prompts.py

is_rejection = any(answer.startswith(t) for t in REJECTION_TAGS)
if is_rejection:
    answer = re.sub(r'^\[(FUORI_SCOPE|KNOWLEDGE_GAP)\]\s*', '', answer)
else:
    answer += self._format_sources(sources)
return answer
```

## Edge Cases

| Case | Behavior |
|---|---|
| LLM omits tag on rejection | Sources shown — degrades to current behavior, acceptable |
| LLM places tag mid-response | Tag not stripped, visible to user — rare, acceptable |
| Stream ends before buffer fills | `tag_checked` remains False, nothing yielded — only possible on near-empty responses |
| `chat()` path (not used by app.py) | Same logic, simpler: full string check + regex strip |

## Files Changed

- `src/prompts.py` — update `SYSTEM_PROMPT` rules 3 & 4, add `REJECTION_TAGS`
- `src/brain.py` — update `chat_stream()` and `chat()`, import `REJECTION_TAGS`
