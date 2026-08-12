import os
import sqlite3
import yaml

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_MEMORY_CONFIG"),
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

class SLMMemoryManager:
    """
    Manages long-term personal state and preference graphs. Learns and adapts to user query
    patterns locally without cloud synchronization.
    """
    def __init__(self, db_path=None):
        self.config, _ = load_config()
        path = db_path or self.config.get("storage", {}).get("db_path", "~/.cache/slm_memory/user_state.db")
        self.db_path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS facts ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "fact TEXT UNIQUE, "
            "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.commit()
        conn.close()

    def store_fact(self, fact: str) -> bool:
        """
        Stores a user fact or preference statement into the local knowledge database.
        """
        if not fact:
            return False
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO facts (fact) VALUES (?)", (fact.strip(),))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def get_relevant_facts(self, query: str = "", system_prompt: str = None, user_input: str = None) -> list[str]:
        """
        Retrieves stored facts matching query keywords.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT fact FROM facts ORDER BY id DESC LIMIT 50")
        rows = cursor.fetchall()
        conn.close()

        all_facts = [r[0] for r in rows]
        if not query:
            return all_facts

        # Simple keyword match filter
        keywords = [w.lower() for w in query.split() if len(w) > 3]
        relevant = [f for f in all_facts if any(kw in f.lower() for kw in keywords)]
        return relevant or all_facts[:3]
