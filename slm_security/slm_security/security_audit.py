import os
import re
import yaml

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_SECURITY_CONFIG"),
        "./config.yaml",
        "../config.yaml",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
    ]
    for path in config_paths:
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return yaml.safe_load(f) or {}, os.path.abspath(path)
            except Exception:
                pass
    return {}, ""

class SLMSecurityAudit:
    """
    A sub-5ms CPU-optimized input/output safety guardrail and PII audit agent for local SLM workflows.
    """
    PII_PATTERNS = {
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    }

    INJECTION_PATTERNS = [
        (r"\b(rm\s+-rf|sudo\s+|chmod\s+777|mkfs|dd\s+if=)\b", "Dangerous Shell Command Detected"),
        (r"\b(DROP\s+TABLE|DELETE\s+FROM|UNION\s+SELECT|ALTER\s+TABLE|INSERT\s+INTO.*VALUES)\b", "SQL Injection Attempt Detected"),
        (r"(ignore\s+previous\s+instructions|system\s+override|jailbreak\s+mode|act\s+as\s+DAN)", "Prompt Jailbreak Attempt Detected")
    ]

    def __init__(self, redact_pii=True, block_injections=True):
        self.config, _ = load_config()
        sec_cfg = self.config.get("security", {})
        self.redact_pii = redact_pii if redact_pii is not None else sec_cfg.get("redact_pii", True)
        self.block_injections = block_injections if block_injections is not None else sec_cfg.get("block_injections", True)

    def sanitize(self, input_text: str) -> dict:
        """
        Sanitizes text by redacting PII and detecting safety or code injection violations.
        Returns a dict: {"safe": bool, "sanitized_text": str, "violations": list[str]}
        """
        if not input_text:
            return {"safe": True, "sanitized_text": "", "violations": []}

        sanitized_text = input_text
        violations = []

        # 1. PII Redaction
        if self.redact_pii:
            for pii_type, pattern in self.PII_PATTERNS.items():
                if re.search(pattern, sanitized_text):
                    sanitized_text = re.sub(pattern, f"[REDACTED_{pii_type}]", sanitized_text)

        # 2. Injection & Safety Checks
        if self.block_injections:
            for pattern, msg in self.INJECTION_PATTERNS:
                if re.search(pattern, input_text, re.IGNORECASE):
                    violations.append(msg)

        is_safe = len(violations) == 0

        return {
            "safe": is_safe,
            "sanitized_text": sanitized_text if is_safe else f"[BLOCKED] Safety Violation: {', '.join(violations)}",
            "violations": violations
        }
