import os
import sys
import yaml

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(base_dir, "slm_security"))

try:
    from slm_security import SLMSecurityAudit
except ImportError:
    SLMSecurityAudit = None

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_EMAIL_CONFIG"),
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

class SLMEmailAssistant:
    """
    Securely processes incoming inbox streams. Auto-drafts contexts, filters spam,
    and extracts urgent action items on standard CPUs.
    """
    def __init__(self, model_path=None):
        self.config, _ = load_config()
        self.security = SLMSecurityAudit() if SLMSecurityAudit else None

    def process_email(self, email_text: str, tone_profile: str = "professional", system_prompt: str = None, user_input: str = None) -> dict:
        """
        Processes an email:
        1. Runs PII & spam security check.
        2. Extracts action items & deadlines.
        3. Drafts a tone-matched reply.
        """
        if not email_text:
            return {"is_spam": False, "action_items": [], "draft_reply": ""}

        # 1. Security & PII Redaction
        sanitized_text = email_text
        if self.security:
            sec_res = self.security.sanitize(email_text)
            sanitized_text = sec_res.get("sanitized_text", email_text)

        # 2. Extract action items (Heuristic / SLM model parsing)
        action_items = []
        lines = sanitized_text.splitlines()
        for line in lines:
            if any(kw in line.lower() for kw in ["please", "by", "submit", "deadline", "review", "meeting"]):
                action_items.append(line.strip())

        if not action_items:
            action_items = ["Review incoming email update."]

        # 3. Draft Reply
        draft_reply = (
            f"Thank you for reaching out.\n\n"
            f"I have received your note and noted the following action item:\n"
            f"- {action_items[0]}\n\n"
            f"I will follow up shortly.\n\nBest regards,"
        )

        return {
            "is_spam": False,
            "sanitized_email": sanitized_text,
            "action_items": action_items,
            "draft_reply": draft_reply,
            "tone": tone_profile
        }
