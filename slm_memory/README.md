# SLM Memory Manager

A CPU-optimized local Memory and User Preference Manager. It handles long-term state tracking, user query adaptations, and keyword-based semantic retrieval using a local SQLite database.

---

## Features

- **SQLite Database Backend**: Persists user facts and settings locally.
- **Preference Retrieval**: Performs queries against the database to fetch relevant background facts for active context generation.
- **Isolate Connections**: Supports custom DB file paths (e.g. in-memory or temp files) to isolate execution tests.

---

## Installation

```bash
pip install -e ./slm_memory
```

---

## API Reference

### `SLMMemoryManager`

```python
from slm_memory import SLMMemoryManager

# Uses default path (~/.cache/slm_memory/user_state.db) or custom path
mem = SLMMemoryManager(db_path=":memory:")
```

#### `store_fact(fact: str) -> bool`
Persists a single preference or context statement.
- **Arguments**:
  - `fact` (str): Statement to remember (e.g. "User prefers Python over JavaScript.").
- **Returns**:
  - `bool`: `True` if successfully stored, `False` otherwise.

#### `get_relevant_facts(query: str) -> list`
Retrieves facts matching keywords inside the query.
- **Arguments**:
  - `query` (str): Keyword lookup string (e.g. "preferences").
- **Returns**:
  - `list`: Array of matched fact strings.

---

## Usage Example

```python
from slm_memory import SLMMemoryManager

mem = SLMMemoryManager(db_path=":memory:")

# Store user facts
mem.store_fact("User prefers Python over JavaScript.")
mem.store_fact("User works on local CPU RAG agents.")

# Retrieve facts
matched = mem.get_relevant_facts("What languages does the user prefer?")
print(matched) # Output: ['User prefers Python over JavaScript.']
```

### Input & Output Example

#### Input (Fact Storage):
`"User prefers python code examples."`

#### Input (Query Lookup):
`"code preferences"`

#### Output:
```json
[
  "User prefers python code examples."
]
```
