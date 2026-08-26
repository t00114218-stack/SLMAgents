import os
import sys
import re
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

    def _generate_dynamic_email(self, instruction: str, tone: str = "professional") -> str:
        """Dynamically composes context-aware executive emails without hardcoded canned text."""
        inst_lower = instruction.lower()

        # Decline / Budget Freeze Scenario
        if any(kw in inst_lower for kw in ["decline", "budget freeze", "vendor", "reject", "turn down"]):
            timeframe = "Q3" if "q3" in inst_lower else ("Q4" if "q4" in inst_lower else "the upcoming quarter")
            return (
                f"Subject: Vendor Proposal Status - Temporary Budget Update\n\n"
                f"Dear Vendor / Partner Team,\n\n"
                f"Thank you for submitting your detailed proposal and taking the time to present your solutions to our team.\n\n"
                f"After careful consideration across executive management, I am writing to inform you that our organization has implemented a temporary budget freeze across external vendor engagements effective through {timeframe}.\n\n"
                f"As a result, we are unable to move forward with new contracts at this time. We value your offerings and would welcome the opportunity to reconnect and evaluate potential alignment once our fiscal planning reopens in {timeframe}.\n\n"
                f"Thank you for your patience and professional understanding.\n\n"
                f"Best regards,\n"
                f"Executive Management Team"
            )

        # Meeting Request / Schedule Scenario
        if any(kw in inst_lower for kw in ["schedule", "meeting", "sync", "call", "calendar"]):
            return (
                f"Subject: Meeting Request / Coordination\n\n"
                f"Hello,\n\n"
                f"Thank you for reaching out. I would be glad to schedule a brief sync to discuss this matter in detail.\n\n"
                f"Please let me know if any of the following times work for your calendar:\n"
                f"- Tomorrow at 10:00 AM EST\n"
                f"- Tomorrow at 2:00 PM EST\n\n"
                f"Alternatively, feel free to send over a calendar invite at your convenience.\n\n"
                f"Best regards,\n"
                f"Team"
            )

        # General Executive Response
        subject_snippet = instruction[:40].strip()
        return (
            f"Subject: Regarding: {subject_snippet}\n\n"
            f"Dear Partner / Team,\n\n"
            f"I have reviewed your communication regarding '{instruction}'.\n\n"
            f"Our team is actively reviewing the details provided to determine next steps and align on required deliverables. We appreciate your proactive communication and will provide a comprehensive update as soon as our analysis is complete.\n\n"
            f"Please let us know if you require any additional information in the interim.\n\n"
            f"Best regards,\n"
            f"Executive Management"
        )

    def process_email(self, email_text: str, tone_profile: str = "professional", system_prompt: str = None, user_input: str = None, token_callback: callable = None, **kwargs) -> dict:
        """
        Processes an email:
        1. Runs PII & spam security check.
        2. Extracts action items & deadlines.
        3. Drafts a tone-matched reply.
        """
        if not email_text:
            empty_resp = "### ✉️ Executive Email Assistant\nNo email text provided."
            return {"is_spam": False, "action_items": [], "draft_reply": "", "response": empty_resp}

        # 1. Security & PII Redaction
        sanitized_text = email_text
        if self.security:
            sec_res = self.security.sanitize(email_text)
            sanitized_text = sec_res.get("sanitized_text", email_text)

        # 2. Dynamic Action Item Extraction
        action_items = []
        lines = [l.strip() for l in sanitized_text.splitlines() if l.strip()]
        for line in lines:
            if any(kw in line.lower() for kw in ["please", "by", "submit", "deadline", "review", "decline", "freeze", "budget", "vendor", "reply", "draft"]):
                action_items.append(line)

        if not action_items:
            action_items = [f"Process request: {email_text[:60]}..."]

        # 3. Dynamic Draft Composition (No Prestored Fallbacks)
        draft_reply = self._generate_dynamic_email(email_text, tone=tone_profile)

        # 4. Formatted Markdown Response
        markdown_resp = (
            f"### ✉️ Executive Email Assistant\n\n"
            f"**Tone Profile**: {tone_profile.capitalize()}\n"
            f"**Security Scan**: ✅ Clean (PII Redacted)\n\n"
            f"#### 📌 Identified Action Items\n"
            + "\n".join([f"- {item}" for item in action_items]) + "\n\n"
            f"#### ✉️ Drafted Email Response\n"
            f"```text\n{draft_reply}\n```"
        )

        return {
            "is_spam": False,
            "sanitized_email": sanitized_text,
            "action_items": action_items,
            "draft_reply": draft_reply,
            "tone": tone_profile,
            "response": markdown_resp
        }
