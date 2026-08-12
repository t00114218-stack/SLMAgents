# SLM Email Assistant

A CPU-optimized local Email Assistant designed to analyze incoming emails, flag spam, extract action items and deadlines, adjust tone profiles, and auto-draft context-appropriate replies.

---

## Features

- **Action Item & Deadline Extraction**: Scans raw email bodies to list specific tasks, requests, and target dates.
- **Spam Filtering**: Local logic to determine if emails are promotional, spam, or noise.
- **Auto-draft Reply Generation**: Drafts polite, professional, or concise replies using the extracted details.
- **100% Offline & Secure**: Protects sensitive inbox communications from cloud data leakage.

---

## Installation

```bash
pip install -e ./slm_email
```

---

## API Reference

### `SLMEmailAssistant`

```python
from slm_email import SLMEmailAssistant

assistant = SLMEmailAssistant(tone="professional")
```

#### `process_email(email_text: str) -> dict`
Analyzes the email text to extract tasks, detect spam, and generate a draft response.
- **Arguments**:
  - `email_text` (str): Raw content body of the incoming email.
- **Returns**:
  - `dict`:
    ```python
    {
        "is_spam": True/False,
        "sanitized_email": str,     # Email content stripped of PII
        "action_items": list,       # Identified action tasks
        "draft_reply": str,         # Suggested auto-draft response
        "tone": str                 # Configured response tone profile
    }
    ```

---

## Usage Example

```python
from slm_email import SLMEmailAssistant

assistant = SLMEmailAssistant(tone="professional")
email_body = "Hi team, please submit the project report by Friday 5 PM."

result = assistant.process_email(email_body)

print(f"Is Spam: {result['is_spam']}")
print(f"Tasks: {result['action_items']}")
print(f"Draft Auto-reply:\n{result['draft_reply']}")
```

### Input & Output Example

#### Input (Email Text):
```text
Hi team, please submit the project report by Friday 5 PM.
```

#### Output:
```json
{
  "is_spam": false,
  "sanitized_email": "Hi team, please submit the project report by Friday 5 PM.",
  "action_items": [
    "Please submit the project report by Friday 5 PM."
  ],
  "draft_reply": "Thank you for reaching out.\n\nI have received your note and noted the following action item:\n- Please submit the project report by Friday 5 PM.\n\nI will follow up shortly.\n\nBest regards,",
  "tone": "professional"
}
```
