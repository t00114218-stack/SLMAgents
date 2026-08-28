import os
import sqlite3
import yaml
import json
import time

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

class SessionState:
    """
    State Graph & Context Object for an active SLMAgents session.
    Maintains vector DB paths, document references, session assets, turn history,
    agent-specific working contexts, and dynamic graph state passed across agents.
    """
    def __init__(self, session_id: str, vector_db_path: str = None):
        self.session_id = session_id or "default_session"
        cache_dir = os.path.expanduser("~/.cache/slm_memory/vector_stores")
        os.makedirs(cache_dir, exist_ok=True)
        self.vector_db_path = vector_db_path or os.path.join(cache_dir, f"{self.session_id}_vector.db")
        self.documents = []        # list of doc dicts
        self.active_document = None
        self.assets = []           # list of asset dicts (images, code files, CSVs, audio)
        self.turns = []            # conversational history
        self.agent_states = {}     # per-agent isolated state dicts {"code_interpreter": {}, "db_migration": {}, ...}
        self.variables = {}        # environment and session variables
        self.active_topic = None
        self.last_agent = None
        self.state_graph = {
            "session_id": self.session_id,
            "vector_db_path": self.vector_db_path,
            "agent_states": self.agent_states,
            "created_at": time.time(),
            "updated_at": time.time()
        }

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "vector_db_path": self.vector_db_path,
            "documents": self.documents,
            "active_document": self.active_document,
            "assets": self.assets,
            "turns": self.turns,
            "agent_states": self.agent_states,
            "variables": self.variables,
            "active_topic": self.active_topic,
            "last_agent": self.last_agent,
            "state_graph": self.state_graph
        }

class SLMMemoryManager:
    """
    Manages long-term personal state and session working memory graphs.
    Maintains vector store paths, document references, session assets, and multi-turn state graphs.
    """
    _session_store: dict[str, SessionState] = {}

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
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS session_metadata ("
            "session_id TEXT PRIMARY KEY, "
            "active_topic TEXT, "
            "active_doc_name TEXT, "
            "active_doc_data TEXT, "
            "vector_db_path TEXT, "
            "agent_states TEXT, "
            "assets_data TEXT, "
            "turns_data TEXT, "
            "variables_data TEXT, "
            "last_agent TEXT, "
            "updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        for col, col_type in [
            ("active_topic", "TEXT"),
            ("active_doc_name", "TEXT"),
            ("active_doc_data", "TEXT"),
            ("vector_db_path", "TEXT"),
            ("agent_states", "TEXT"),
            ("assets_data", "TEXT"),
            ("turns_data", "TEXT"),
            ("variables_data", "TEXT"),
            ("last_agent", "TEXT"),
            ("updated_at", "DATETIME DEFAULT CURRENT_TIMESTAMP")
        ]:
            try:
                cursor.execute(f"ALTER TABLE session_metadata ADD COLUMN {col} {col_type}")
            except Exception:
                pass
        conn.commit()
        conn.close()

    def get_or_create_session(self, session_id: str) -> SessionState:
        """Retrieves or creates the SessionState graph object for the given session ID."""
        if not session_id:
            session_id = "default_session"
        if session_id not in self._session_store:
            session = SessionState(session_id)
            # Restore state for this specific session from SQLite database if available
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT active_doc_data, vector_db_path, agent_states, assets_data, turns_data, variables_data, last_agent FROM session_metadata WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                conn.close()
                if row:
                    active_doc_data, v_path, a_states, a_assets, t_turns, v_vars, l_agent = row
                    if active_doc_data:
                        doc_info = json.loads(active_doc_data)
                        session.active_document = doc_info
                        session.documents.append(doc_info)
                    if v_path:
                        session.vector_db_path = v_path
                    if a_states:
                        session.agent_states = json.loads(a_states)
                    if a_assets:
                        session.assets = json.loads(a_assets)
                    if t_turns:
                        session.turns = json.loads(t_turns)
                    if v_vars:
                        session.variables = json.loads(v_vars)
                    if l_agent:
                        session.last_agent = l_agent
            except Exception:
                pass
            self._session_store[session_id] = session
        return self._session_store[session_id]


    def clear_session(self, session_id: str) -> bool:
        """Completely clears and resets all memory, documents, and context for a specific session."""
        if not session_id:
            session_id = "default_session"
        if session_id in self._session_store:
            session = self._session_store[session_id]
            if session.vector_db_path and os.path.exists(session.vector_db_path):
                try:
                    os.remove(session.vector_db_path)
                except Exception:
                    pass
            del self._session_store[session_id]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM session_metadata WHERE session_id = ?", (session_id,))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def clear_all(self) -> bool:
        """Clears all session memories, documents, and active contexts globally."""
        for session in self._session_store.values():
            if session.vector_db_path and os.path.exists(session.vector_db_path):
                try:
                    os.remove(session.vector_db_path)
                except Exception:
                    pass
        self._session_store.clear()
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM session_metadata")
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def set_vector_db_path(self, session_id: str, vector_db_path: str):
        """Sets and persists the vector database file path for a session."""
        session = self.get_or_create_session(session_id)
        session.vector_db_path = vector_db_path
        session.state_graph["vector_db_path"] = vector_db_path
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO session_metadata (session_id, vector_db_path) VALUES (?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET vector_db_path=excluded.vector_db_path",
                (session_id, vector_db_path)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_vector_db_path(self, session_id: str) -> str:
        """Retrieves the vector database path for a session."""
        session = self.get_or_create_session(session_id)
        return session.vector_db_path

    def store_fact(self, fact: str) -> bool:
        """Stores a user fact or preference statement into local database."""
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
        """Retrieves stored facts matching query keywords."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT fact FROM facts ORDER BY id DESC LIMIT 50")
            rows = cursor.fetchall()
            conn.close()
            all_facts = [r[0] for r in rows]
        except Exception:
            all_facts = []

        if not query:
            return all_facts

        keywords = [w.lower() for w in query.split() if len(w) > 2]
        relevant = [f for f in all_facts if any(kw in f.lower() for kw in keywords)]
        return relevant if relevant else all_facts

    # --- Session Working Memory, Document & Asset State Graph ---
    def store_document_memory(self, session_id: str, doc_name: str, chunks: list[str] = None, summary: str = None, file_path: str = None, vector_db_path: str = None, full_text: str = None, is_in_memory_direct: bool = True):
        """Stores document metadata, in-memory full text, vector DB path, and text chunks into session state graph and SQLite."""
        session = self.get_or_create_session(session_id)
        
        v_path = vector_db_path or session.vector_db_path
        session.vector_db_path = v_path

        chunks = chunks or ([full_text] if full_text else [])
        total_chars = len(full_text) if full_text else sum(len(c) for c in chunks)

        doc_info = {
            "name": doc_name,
            "path": file_path,
            "vector_db_path": v_path,
            "full_text": full_text or ("\n\n".join(chunks) if chunks else ""),
            "chunks": chunks,
            "summary": summary,
            "is_in_memory_direct": is_in_memory_direct,
            "character_count": total_chars,
            "timestamp": time.time()
        }

        session.active_document = doc_info
        session.documents.append(doc_info)
        session.last_agent = "SLMRag"
        session.state_graph["active_document"] = doc_info
        session.state_graph["vector_db_path"] = v_path
        session.state_graph["updated_at"] = time.time()

        # Persist active document to SQLite database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO session_metadata (session_id, active_doc_name, active_doc_data, vector_db_path, last_agent, updated_at) "
                "VALUES (?, ?, ?, ?, 'SLMRag', CURRENT_TIMESTAMP) "
                "ON CONFLICT(session_id) DO UPDATE SET active_doc_name=excluded.active_doc_name, active_doc_data=excluded.active_doc_data, vector_db_path=excluded.vector_db_path, last_agent='SLMRag', updated_at=CURRENT_TIMESTAMP",
                (session_id or "default_session", doc_name, json.dumps(doc_info), v_path)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[SLMMemoryManager] SQLite store note: {e}")

    def set_agent_state(self, session_id: str, agent_name: str, state_data: dict):
        """Sets and persists agent-specific isolated working state for this session."""
        session = self.get_or_create_session(session_id)
        session.agent_states[agent_name] = state_data or {}
        session.state_graph["agent_states"] = session.agent_states
        session.state_graph["updated_at"] = time.time()
        self._sync_session_to_db(session)

    def get_agent_state(self, session_id: str, agent_name: str) -> dict:
        """Retrieves agent-specific isolated working state for this session."""
        session = self.get_or_create_session(session_id)
        return session.agent_states.get(agent_name, {})

    def update_agent_state(self, session_id: str, agent_name: str, updates: dict) -> dict:
        """Updates agent-specific isolated state variables for this session."""
        session = self.get_or_create_session(session_id)
        if agent_name not in session.agent_states:
            session.agent_states[agent_name] = {}
        session.agent_states[agent_name].update(updates or {})
        session.state_graph["agent_states"] = session.agent_states
        session.state_graph["updated_at"] = time.time()
        self._sync_session_to_db(session)
        return session.agent_states[agent_name]

    def clear_agent_state(self, session_id: str, agent_name: str):
        """Clears agent-specific isolated state for this session."""
        session = self.get_or_create_session(session_id)
        if agent_name in session.agent_states:
            del session.agent_states[agent_name]
            session.state_graph["agent_states"] = session.agent_states
            self._sync_session_to_db(session)

    def add_asset(self, session_id: str, asset_type: str, file_path: str, metadata: dict = None):
        """Registers a session asset (image, code output, PDF, audio) into session state graph."""
        session = self.get_or_create_session(session_id)
        asset_info = {
            "type": asset_type,
            "path": file_path,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        session.assets.append(asset_info)
        session.state_graph["assets"] = session.assets
        self._sync_session_to_db(session)

    def get_assets(self, session_id: str, asset_type: str = None) -> list[dict]:
        """Retrieves all assets registered in this session, optionally filtered by type."""
        session = self.get_or_create_session(session_id)
        if asset_type:
            return [a for a in session.assets if a.get("type") == asset_type]
        return list(session.assets)

    def update_state_graph(self, session_id: str, updates: dict):
        """Updates arbitrary state variables on the session's state graph."""
        session = self.get_or_create_session(session_id)
        session.state_graph.update(updates)
        session.state_graph["updated_at"] = time.time()
        self._sync_session_to_db(session)

    def get_active_document(self, session_id: str = None) -> dict | None:
        """Retrieves the active document working context for a session if one exists."""
        session = self.get_or_create_session(session_id or "default_session")
        return session.active_document

    def _sync_session_to_db(self, session: SessionState):
        """Persists full session state graph to SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO session_metadata (session_id, active_topic, active_doc_name, active_doc_data, vector_db_path, agent_states, assets_data, turns_data, variables_data, last_agent, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "active_topic=excluded.active_topic, "
                "active_doc_name=excluded.active_doc_name, "
                "active_doc_data=excluded.active_doc_data, "
                "vector_db_path=excluded.vector_db_path, "
                "agent_states=excluded.agent_states, "
                "assets_data=excluded.assets_data, "
                "turns_data=excluded.turns_data, "
                "variables_data=excluded.variables_data, "
                "last_agent=excluded.last_agent, "
                "updated_at=CURRENT_TIMESTAMP",
                (
                    session.session_id,
                    session.active_topic,
                    session.active_document.get("name") if session.active_document else None,
                    json.dumps(session.active_document) if session.active_document else None,
                    session.vector_db_path,
                    json.dumps(session.agent_states),
                    json.dumps(session.assets),
                    json.dumps(session.turns),
                    json.dumps(session.variables),
                    session.last_agent
                )
            )
            conn.commit()
            conn.close()
        except Exception as e:
            pass


    def extract_memory_facts_with_phi(self, user_text: str) -> list[str]:
        """Uses the Phi 4B ONNX engine to extract structured long-term facts/preferences from user input."""
        if not user_text or len(user_text.split()) < 3:
            return []
        
        # Pre-filter for explicit fact/preference indicators
        memory_keywords = ["remember", "my name", "i like", "i prefer", "my email", "my database", "my company", "always use", "never use", "favourite", "favorite", "our stack", "my timezone"]
        if not any(kw in user_text.lower() for kw in memory_keywords):
            return []

        try:
            import sys
            main_mod = sys.modules.get("main")
            if main_mod and hasattr(main_mod, "get_shared_orchestrator"):
                orchestrator = main_mod.get_shared_orchestrator()
                if orchestrator and hasattr(orchestrator, "model") and orchestrator.model:
                    prompt = (
                        "<|im_start|>system\n"
                        "Extract key personal facts, preferences, or technical specifications from the user text. "
                        "Return ONLY bullet points starting with '- '. If no long-term fact is present, output 'None'.<|im_end|>\n"
                        f"<|im_start|>user\n{user_text}<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    )
                    import onnxruntime_genai as og
                    input_tokens = orchestrator.tokenizer.encode(prompt)
                    params = og.GeneratorParams(orchestrator.model)
                    params.set_search_options(max_length=len(input_tokens) + 48, temperature=0.1)
                    generator = og.Generator(orchestrator.model, params)
                    generator.append_tokens(input_tokens)
                    output_tokens = []
                    while not generator.is_done():
                        generator.generate_next_token()
                        new_tokens = generator.get_next_tokens()
                        if len(new_tokens) > 0:
                            tok_id = int(new_tokens[0])
                            if tok_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                                break
                            output_tokens.append(tok_id)
                    raw_out = orchestrator.tokenizer.decode(output_tokens).strip()
                    facts = [line.lstrip("- ").strip() for line in raw_out.splitlines() if line.strip().startswith("- ") and len(line.strip()) > 3]
                    for f in facts:
                        self.store_fact(f)
                    return facts
        except Exception as e:
            print(f"[SLMMemoryManager] Phi fact extraction note: {e}")

        # Fallback fact storage
        self.store_fact(user_text.strip())
        return [user_text.strip()]

    def record_turn(self, session_id: str, user_text: str, assistant_text: str, agent: str = None):
        """Records a conversational turn in session working memory and extracts long-term facts via Phi 4B."""
        session = self.get_or_create_session(session_id)
        turn = {
            "user": user_text,
            "assistant": assistant_text,
            "agent": agent,
            "timestamp": time.time()
        }
        session.turns.append(turn)
        if len(session.turns) > 15:
            session.turns = session.turns[-15:]
        if agent:
            session.last_agent = agent

        # Extract long-term facts via Phi 4B engine
        self.extract_memory_facts_with_phi(user_text)

    def get_session_history(self, session_id: str) -> list[dict]:
        """Retrieves recent conversation turns for context resolution."""
        session = self.get_or_create_session(session_id)
        return session.turns

    def resolve_context(self, session_id: str, query: str, history: list[dict] = None) -> dict:
        """
        Analyzes the query against active session state graph to determine the context
        and decide what assets, vector DB path, document chunks, and history should be passed to agents.
        """
        session = self.get_or_create_session(session_id)
        active_doc = session.active_document
        v_path = session.vector_db_path
        q_lower = (query or "").lower().strip()

        # Check explicit intent overrides (only override if explicitly requesting code execution or SQL)
        non_doc_intents = ["run python", "write python", "execute code", "generate sql", "write sql", "translate to", "send email", "solve equation"]
        is_explicit_other = any(intent in q_lower for intent in non_doc_intents)
        
        # Any query while a document context is active routes to SLMRag
        is_doc_followup = bool(active_doc) and not is_explicit_other

        return {
            "session_id": session.session_id,
            "vector_db_path": v_path,
            "has_active_document": bool(active_doc),
            "active_document": active_doc,
            "documents": session.documents,
            "assets": session.assets,
            "is_doc_followup": is_doc_followup,
            "suggested_agent": "SLMRag" if is_doc_followup else None,
            "state_graph": session.state_graph,
            "recent_turns": session.turns[-5:] if session.turns else []
        }
