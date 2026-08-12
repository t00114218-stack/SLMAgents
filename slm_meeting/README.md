# SLM Meeting Summarizer

A local CPU-optimized Meeting Transcript Summarizer. It parses conversation logs, maps meeting participant interactions, highlights key decisions, and compiles task lists into structured Markdown tables.

---

## Features

- **Speaker Mapping**: Resolves conversational dialogue turns to map active participants.
- **Action Item Markdown Table**: Automatically compiles task assignments, owners, and deadlines into clean tables.
- **Decision Highlights**: Summarizes consensus points and meeting outcomes.
- **Offline Security**: Keeps conversational and corporate logs entirely local to prevent data breaches.

---

## Installation

```bash
pip install -e ./slm_meeting
```

---

## API Reference

### `SLMMeetingSummarizer`

```python
from slm_meeting import SLMMeetingSummarizer

summarizer = SLMMeetingSummarizer()
```

#### `summarize_transcript(transcript: str) -> dict`
Processes the dialogue turns to produce summaries, decisions, and action tables.
- **Arguments**:
  - `transcript` (str): Speaker transcript dialogue turns (e.g. "Alice: I will fix the bug.").
- **Returns**:
  - `dict`:
    ```python
    {
        "speakers": list,           # List of resolved speakers
        "decisions": list,          # Highlights of agreements/decisions
        "action_table": str,        # Markdown action table
        "summary": str              # Conversational paragraph summary
    }
    ```

---

## Usage Example

```python
from slm_meeting import SLMMeetingSummarizer

summarizer = SLMMeetingSummarizer()
transcript = "Alice: I will fix the bug by tomorrow. Bob: I will test the schema updates."

result = summarizer.summarize_transcript(transcript)

print(f"Speakers: {result['speakers']}")
print(f"Summary: {result['summary']}")
print(f"Action Table:\n{result['action_table']}")
```

### Input & Output Example

#### Input (Dialogue Transcript):
```text
Alice: I will verify index 8.
```

#### Output:
```json
{
  "speakers": ["Alice"],
  "decisions": [
    "Agreed to follow up on project timeline."
  ],
  "action_table": "| Speaker | Assigned Action Item | Deadline |\n| :--- | :--- | :--- |\n| Alice | I will verify index 8. | TBD |",
  "summary": "Meeting focused on project goals with 1 participant(s)."
}
```
