# SLM Security Audit

A CPU-optimized local Security Guardrail and PII Audit agent. It scans queries and outputs for Personally Identifiable Information (SSN, credit card, API keys), applies redactions, and filters out prompt injection/jailbreak patterns.

---

## Features

- **PII Redaction**: Regular expressions for SSN, credit cards, emails, and phone numbers.
- **Guardrail Filter**: Flags injection keywords (`ignore prior instructions`, `override security`, etc.).
- **Zero Cloud Leaks**: Keeps compliance audits completely internal.

---

## Installation

```bash
pip install -e ./slm_security
```

---

## API Reference

### `SLMSecurityAudit`

```python
from slm_security import SLMSecurityAudit

auditor = SLMSecurityAudit()
```

#### `sanitize(text: str) -> dict`
Scans and redacts sensitive strings or flags violations.
- **Arguments**:
  - `text` (str): Raw string content.
- **Returns**:
  - `dict`:
    ```python
    {
        "safe": True/False,
        "sanitized_text": str,     # Redacted string output
        "violations": list         # Names of triggered security rules
    }
    ```

---

## Usage Example

```python
from slm_security import SLMSecurityAudit

auditor = SLMSecurityAudit()
dirty_input = "My SSN is 000-11-2222 and my email is test@email.com"

result = auditor.sanitize(dirty_input)

print(f"Safe: {result['safe']}")
print(f"Sanitized: {result['sanitized_text']}")
```

### Input & Output Example

#### Input (Dirty String):
```text
SSN is 000-11-2222
```

#### Output:
```json
{
  "safe": true,
  "sanitized_text": "SSN is [REDACTED_SSN]",
  "violations": []
}
```
