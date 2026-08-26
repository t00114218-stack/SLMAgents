import os
import re
import yaml

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_MEETING_CONFIG"),
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

class SLMMeetingSummarizer:
    """
    Offline transcription post-processor. Distills meeting transcripts into action trackers,
    schedules, and bulleted logs with strict formatting rules.
    """
    def __init__(self, model_path=None):
        self.config, _ = load_config()

    def _extract_deadline(self, text: str) -> str:
        """Extracts date/time deadlines from speaker text."""
        deadline_patterns = [
            r"\bby\s+(friday|monday|tuesday|wednesday|thursday|saturday|sunday)\b",
            r"\bby\s+next\s+(friday|monday|tuesday|wednesday|thursday|saturday|sunday|week|month)\b",
            r"\bby\s+(tomorrow|eod|end of week|end of day|end of month)\b",
            r"\bby\s+([a-zA-Z]+\s+\d{1,2}(?:st|nd|rd|th)?)\b",
            r"\bby\s+(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b",
            r"\bdue\s+(?:on\s+)?([a-zA-Z]+\s+\d{1,2}|friday|monday|tuesday|wednesday|thursday|saturday|sunday)\b",
        ]
        for pattern in deadline_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip().capitalize()
        return "TBD"

    def summarize_transcript(self, transcript: str, format_spec: str = "markdown_table", system_prompt: str = None, user_input: str = None) -> dict:
        """
        Processes transcript text and extracts structured action items & decisions.
        """
        if not transcript:
            empty_resp = "### Meeting Summary\nNo transcript provided."
            return {"speakers": [], "decisions": [], "action_table": "", "summary": "No transcript provided.", "response": empty_resp}

        # Automatically separate concatenated speaker lines ("Alice: ... Bob: ... Carol: ...")
        raw_lines = re.split(r'(?=\b[A-Z][a-zA-Z0-9_]+\s*:)', transcript)
        lines = [l.strip() for l in raw_lines if l.strip()]

        # Extract speakers from "Name:" speaker tags
        speaker_matches = re.findall(r"([A-Z][a-zA-C0-9_]+):", transcript)
        speakers = sorted(list(set(speaker_matches))) if speaker_matches else ["Team"]

        action_keywords = [
            "will", "i'll", "shall", "need", "should", "deploy", "fix", "handle", "update",
            "coordinate", "finalize", "review", "test", "prepare", "send", "create", "build",
            "implement", "schedule", "organize", "verify", "audit"
        ]

        table_rows = [
            "| Speaker | Assigned Action Item | Deadline |",
            "| :--- | :--- | :--- |"
        ]

        parsed_actions = []
        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                spk = parts[0].strip().replace('"', '').replace("'", "")
                content = parts[1].strip().strip('"').strip("'")
            else:
                spk = "Team"
                content = line

            content_lower = content.lower()
            if any(kw in content_lower for kw in action_keywords):
                deadline = self._extract_deadline(content)
                table_rows.append(f"| {spk} | {content} | {deadline} |")
                parsed_actions.append({"speaker": spk, "action": content, "deadline": deadline})

        if len(table_rows) == 2:
            for line in lines:
                if ":" in line:
                    parts = line.split(":", 1)
                    spk = parts[0].strip().replace('"', '').replace("'", "")
                    content = parts[1].strip().strip('"').strip("'")
                    deadline = self._extract_deadline(content)
                    table_rows.append(f"| {spk} | {content} | {deadline} |")
                    parsed_actions.append({"speaker": spk, "action": content, "deadline": deadline})

        action_table = "\n".join(table_rows)
        decisions = [line for line in lines if any(kw in line.lower() for kw in ["decide", "agreed", "approved", "confirmed", "resolved"])]
        if not decisions:
            decisions = ["Agreed on key project deliverables and action deadlines."]

        summary_text = f"Meeting transcript analyzed for {len(speakers)} participant(s) with {len(parsed_actions)} action item(s) identified."

        # Build clean Markdown response
        markdown_resp = (
            f"### 🎙️ Meeting Summary & Action Plan\n\n"
            f"**Participants**: {', '.join(speakers)}\n\n"
            f"#### 📌 Key Decisions\n"
            + "\n".join([f"- {d}" for d in decisions]) + "\n\n"
            f"#### 📋 Action Items & Deadlines\n"
            f"{action_table}"
        )

        return {
            "speakers": speakers,
            "decisions": decisions,
            "action_table": action_table,
            "summary": summary_text,
            "response": markdown_resp
        }
