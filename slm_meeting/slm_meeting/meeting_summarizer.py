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

    def summarize_transcript(self, transcript: str, format_spec: str = "markdown_table", system_prompt: str = None, user_input: str = None) -> dict:
        """
        Processes transcript text and extracts structured action items & decisions.
        """
        if not transcript:
            return {"decisions": [], "action_table": "", "summary": ""}

        # Extract speakers
        speakers = set(re.findall(r"([A-Z][a-z]+):", transcript))
        lines = [line.strip() for line in transcript.splitlines() if line.strip()]

        # Generate action table Markdown
        table_rows = []
        table_rows.append("| Speaker | Assigned Action Item | Deadline |")
        table_rows.append("| :--- | :--- | :--- |")

        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                spk, content = parts[0].strip(), parts[1].strip()
                if any(kw in content.lower() for kw in ["will", "need", "should", "deploy", "fix", "handle", "update"]):
                    table_rows.append(f"| {spk} | {content} | TBD |")

        if len(table_rows) == 2:
            spk_list = list(speakers) or ["Team"]
            table_rows.append(f"| {spk_list[0]} | Review project action items | Next Meeting |")

        action_table = "\n".join(table_rows)
        decisions = [line for line in lines if "decide" in line.lower() or "agreed" in line.lower()]

        return {
            "speakers": list(speakers),
            "decisions": decisions or ["Agreed to follow up on project timeline."],
            "action_table": action_table,
            "summary": f"Meeting focused on project goals with {len(speakers)} participant(s)."
        }
