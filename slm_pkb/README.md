# SLM PKB Agent

A local, offline Personal Knowledge Base (PKB) Agent designed to index directories of Markdown files (e.g. Obsidian vaults), analyze note semantic relevance, and suggest links between related topics.

---

## Features

- **Obsidian Vault Indexing**: Scans and parses metadata from directories containing Markdown note files.
- **Link Recommendation Heuristic**: Uses keyword matching and metadata intersections to recommend connections between notes.
- **Graph Link Mapping**: Resolves relations to return structured link arrays.

---

## Installation

```bash
pip install -e ./slm_pkb
```

---

## API Reference

### `SLMPKBAgent`

```python
from slm_pkb import SLMPKBAgent

agent = SLMPKBAgent()
```

#### `index_vault(vault_path: str) -> dict`
Indexes all notes in the directory and suggests link connections.
- **Arguments**:
  - `vault_path` (str): Absolute file directory path to the note vault.
- **Returns**:
  - `dict`:
    ```python
    {
        "success": True/False,
        "vault_path": str,          # Directory path resolved
        "notes_indexed": int,       # Number of markdown files indexed
        "suggested_links": list     # List of link suggestions containing from/to keys
    }
    ```

---

## Usage Example

```python
from slm_pkb import SLMPKBAgent

agent = SLMPKBAgent()
vault = "~/MyObsidianNotes"

result = agent.index_vault(vault)

print(f"Notes Indexed: {result['notes_indexed']}")
print(f"Suggested Links: {result['suggested_links']}")
```

### Input & Output Example

#### Input (Vault Directory):
`"/Users/revathysuryaprakash/Documents/SLMAgents"`

#### Output:
```json
{
  "success": true,
  "vault_path": "/Users/revathysuryaprakash/Documents/SLMAgents",
  "notes_indexed": 175,
  "suggested_links": [
    {
      "from": "agent_design_proposals",
      "to": "accuracy_report",
      "reason": "High semantic similarity match"
    }
  ]
}
```
