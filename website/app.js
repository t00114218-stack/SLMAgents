// Tab switching logic for code panels
function switchTab(btn, targetId) {
  const tabContainer = btn.parentElement;
  const buttons = tabContainer.querySelectorAll('.code-tab-btn');
  buttons.forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  
  const panel = tabContainer.parentElement;
  const contents = panel.querySelectorAll('.code-content');
  contents.forEach(c => c.style.display = 'none');
  
  const target = panel.querySelector(`#${targetId}`);
  if (target) {
    target.style.display = 'block';
  }
}

// Database of 22 Upcoming Agents
const UPCOMING_AGENTS = {
  "database_migrator": {
    name: "SLM Database Migrator",
    category: "Developer Tools",
    catClass: "badge-dev",
    stage: "Verified (CPU)",
    desc: "Analyzes legacy database schemas and generates zero-downtime, CPU-optimized migrations and modern ORM model definitions offline.",
    features: [
      "Direct SQL table schema analysis and dependency mapping",
      "Automatic compatibility matching for migrations",
      "Generates modern SQLAlchemy and Django ORM models",
      "Suggests structural indexing plans for performance improvement"
    ],
    code: "from slm_db_migration import SLMDBMigrator\n\nmigrator = SLMDBMigrator()\nmigration_sql = migrator.generate_migration(from_schema, to_schema)\nprint(migration_sql)",
    input_output: "→ INPUT (To-Schema):\nCREATE TABLE users (id INT PRIMARY KEY, name TEXT, email TEXT);\n\n← OUTPUT:\n{\n  'migration_sql': 'ALTER TABLE users ADD COLUMN email TEXT;',\n  'sandbox_result': 'Migration verified successfully in SQLite sandbox.'\n}"
  },
  "email_assistant": {
    name: "SLM Email Assistant",
    category: "Productivity",
    catClass: "badge-prod",
    stage: "Verified (CPU)",
    desc: "Securely processes your incoming inbox streams. Auto-drafts contexts, filters spam, and extracts urgent action items on standard CPUs.",
    features: [
      "Offline spam classifier and classification tagging",
      "Action item extraction and scheduled task planning",
      "Generates contextual email replies matching your custom tone profile",
      "PII protection — zero emails ever leave your machine"
    ],
    code: "from slm_email import SLMEmailAssistant\n\nassistant = SLMEmailAssistant()\nreply = assistant.process_email(email_text)\nprint(reply)",
    input_output: "→ INPUT:\n\"Please submit the report by Friday.\"\n\n← OUTPUT:\n{\n  'is_spam': False,\n  'action_items': ['Please submit the report by Friday.']\n}"
  },
  "meeting_summarizer": {
    name: "SLM Meeting Summarizer",
    category: "Productivity",
    catClass: "badge-prod",
    stage: "Verified (CPU)",
    desc: "Offline transcription post-processor. Distills meeting transcripts into action trackers, schedules, and bulleted logs with strict formatting rules.",
    features: [
      "Turns conversational text blocks into formal action tables",
      "Identifies speaker intent, decisions, and deadlines",
      "Map-Reduce pipeline support for 2-hour long transcription logs",
      "Strict template outputs matching markdown specifications"
    ],
    code: "from slm_meeting import SLMMeetingSummarizer\n\nsummarizer = SLMMeetingSummarizer()\ntodos = summarizer.summarize_transcript(transcript_text)\nprint(todos)",
    input_output: "→ INPUT:\n\"Alice: I will deploy the schema.\"\n\n← OUTPUT:\n{\n  'speakers': ['Alice'],\n  'action_table': '| Speaker | Assigned Action Item | Deadline |\\n| Alice | I will deploy the schema. | TBD |'\n}"
  },
  "voice_agent": {
    name: "SLM Voice Agent",
    category: "Productivity",
    catClass: "badge-prod",
    stage: "Verified (CPU)",
    desc: "Fast offline conversational companion combining Whisper speech-to-text, 1.5B chat, and lightweight text-to-speech pipelines on CPU.",
    features: [
      "Offline audio-to-text speech transcription",
      "Low-latency response generation using quantized ONNX",
      "Text-to-speech synthesis utilizing local CPU synthesizer models",
      "Hands-free voice trigger support"
    ],
    code: "from slm_voice import SLMVoiceAgent\n\nvoice = SLMVoiceAgent()\nvoice.process_speech_text(\"Hello local CPU assistant\")",
    input_output: "→ INPUT:\n\"Hello local CPU assistant\"\n\n← OUTPUT:\n{\n  'transcript': 'Hello local CPU assistant',\n  'response': \"I heard you ask: 'Hello local CPU assistant'. Processing your query locally on CPU.\",\n  'audio_synthesized': False\n}"
  },
  "memory_manager": {
    name: "SLM Memory Manager",
    category: "Productivity",
    catClass: "badge-prod",
    stage: "Verified (CPU)",
    desc: "Manages long-term personal state and preference graphs. Learns and adapts to user query patterns locally without cloud synchronization.",
    features: [
      "Entities and relations extraction from chat history",
      "Builds a local knowledge graph of user preferences",
      "Prunes older irrelevant details to fit within context limits",
      "Auto-injects user context tags into RAG sessions"
    ],
    code: "from slm_memory import SLMMemoryManager\n\nmem = SLMMemoryManager()\nmem.store_fact(\"User prefers python code examples.\")\nprint(mem.get_relevant_facts(\"code preferences\"))",
    input_output: "→ INPUT (Store Fact):\n\"User prefers python code examples.\"\n\n← OUTPUT (Fact Retrieval):\n[\n  'User prefers python code examples.'\n]"
  },
  "task_planner": {
    name: "SLM Task Planner",
    category: "Productivity",
    catClass: "badge-prod",
    stage: "Verified (CPU)",
    desc: "Autonomous goal decomposition system. Breaks complex tasks into prioritized action items and assigns them to specialized local sub-agents.",
    features: [
      "Goal decomposition and sub-task scheduling",
      "Dependency mapping for parallel execution branches",
      "Runtime execution tracker with dynamic adjustment",
      "Fallback handler to revise tasks if a sub-agent fails"
    ],
    code: "from slm_task_planner import SLMTaskPlanner\n\nplanner = SLMTaskPlanner()\nplan = planner.build_plan(\"Extract stats 1 from PDF\")\nprint(plan)",
    input_output: "→ INPUT (Goal):\n\"Extract stats 1 from PDF\"\n\n← OUTPUT (Plan):\n{\n  'goal': 'Extract stats 1 from PDF',\n  'tasks': [{'step': 1, 'task': 'Extract layout & tabular data from document', 'assigned_agent': 'SLMPDFChat / SLMDocumentParser'}],\n  'total_steps': 1\n}"
  },
  "pdf_chat": {
    name: "SLM PDF Chat",
    category: "Productivity",
    catClass: "badge-prod",
    stage: "Verified (CPU)",
    desc: "Securely parses complex PDF documents. Assembles layouts, reads tables, and lets you chat with local legal contracts, research articles, or receipts.",
    features: [
      "Locally extracts layout text and multi-column paragraphs",
      "Parses database tables inside PDFs directly to list-of-dicts",
      "Built-in RAG chunk generator for offline querying",
      "Supports scanned image PDFs via local OCR integration"
    ],
    code: "from slm_pdf import SLMPDFChat\n\npdf = SLMPDFChat()\npdf.load(\"invoice.pdf\")\nans = pdf.ask(\"What is the total due amount?\")\nprint(ans)",
    input_output: "→ INPUT (Ask before load):\n\"What is total revenue?\"\n\n← OUTPUT:\n\"No PDF document loaded. Please call `.load(pdf_path)` first.\""
  },
  "pkb_agent": {
    name: "SLM PKB Agent",
    category: "Productivity",
    catClass: "badge-prod",
    stage: "Verified (CPU)",
    desc: "Local knowledge management assistant. Builds, links, and tags markdown documents in Obsidian, Notion, or Logseq vaults offline.",
    features: [
      "Auto-scans directories of markdown notes to map semantic clusters",
      "Suggests links between notes based on context similarity",
      "Auto-generates summaries, tags, and indexing logs for vault folders",
      "Integrates directly with local Obsidian vaults"
    ],
    code: "from slm_pkb import SLMPKBAgent\n\nagent = SLMPKBAgent()\nprint(agent.index_vault(\"~/Obsidian/MyVault\"))",
    input_output: "→ INPUT (Vault Path):\n\"~/MyObsidianVault\"\n\n← OUTPUT:\n{\n  'notes_indexed': 0,\n  'suggested_links': []\n}"
  },
  "data_analyst": {
    name: "SLM Data Analyst",
    category: "Data & Utilities",
    catClass: "badge-data",
    stage: "Verified (CPU)",
    desc: "Loads local CSV, Parquet, or Excel files. Answers statistical questions, performs calculations, and auto-generates data visualization code.",
    features: [
      "Direct pandas dataframe parsing and stats calculator",
      "Translates user query into python matplotlib/pandas code blocks",
      "Generates summary tables and column distribution charts",
      "100% offline analysis of highly sensitive company sheets"
    ],
    code: "from slm_data import SLMDataAnalyst\n\nanalyst = SLMDataAnalyst()\nresult = analyst.analyze_file(\"sales.csv\", \"summarize sales\")\nprint(result)",
    input_output: "→ INPUT (CSV):\n{\"file\": \"sales.csv\", \"query\": \"summarize sales\"}\n\n← OUTPUT:\n{\n  'columns': [],\n  'summary': 'Calculated total revenue by region: East ($15,000), West ($22,000).'\n}"
  },
  "translation_hub": {
    name: "SLM Translation Hub",
    category: "Data & Utilities",
    catClass: "badge-data",
    stage: "Verified (CPU)",
    desc: "Quantized multilingual translation library designed for offline local document conversion across 20+ language profiles.",
    features: [
      "Quantized translation weights optimized for CPU RAM footprint",
      "Preserves original formatting (HTML, Markdown, DOCX markup)",
      "Sentence-alignment validation for precise paragraph mappings",
      "Completely offline operation — ideal for restricted documents"
    ],
    code: "from slm_translation import SLMTranslationHub\n\nhub = SLMTranslationHub()\ntranslated = hub.translate(\"hello world\", source_lang=\"en\", target_lang=\"hi\")\nprint(translated)",
    input_output: "→ INPUT (En -> Hi):\n\"hello world\"\n\n← OUTPUT:\n\"नमस्ते दुनिया\""
  },
  "math_agent": {
    name: "SLM Math Agent",
    category: "Data & Utilities",
    catClass: "badge-data",
    stage: "Verified (CPU)",
    desc: "Specialized arithmetic reasoning model. Handles math formulations, algebraic simplifications, and steps through complex equations offline.",
    features: [
      "Symbolic algebra calculator mapping using local SymPy",
      "Parses equations and graphs steps to final result",
      "Verifies intermediate steps to prevent math hallucinations",
      "Optimized math tokens prompt training templates"
    ],
    code: "from slm_math import SLMMathAgent\n\nagent = SLMMathAgent()\nsteps = agent.solve(\"integrate x^2 from 0 to 3\")\nprint(steps)",
    input_output: "→ INPUT:\n\"integrate x^2 from 0 to 3\"\n\n← OUTPUT:\n{\n  'equation': 'integrate(x^2, 0, 3)',\n  'result': '9'\n}"
  },
  "vision_parser": {
    name: "SLM Vision Parser",
    category: "Data & Utilities",
    catClass: "badge-data",
    stage: "Verified (CPU)",
    desc: "Offline chart, diagram, and whiteboard reader. Converts scanned infographics and drawings to clean structured text summaries.",
    features: [
      "Quantized local Vision-Language model (VLM) weights",
      "Extracts key numbers and trends from bar, line, and pie charts",
      "OCR reader for whiteboards and handwritten flowcharts",
      "Translates infographics directly to clean markdown tables"
    ],
    code: "from slm_vision_parser.vision_parser import SLMVisionParser\n\nparser = SLMVisionParser()\nchart_info = parser.parse_image(\"chart_8.png\", \"<OCR>\")\nprint(chart_info)",
    input_output: "→ INPUT:\n{\"image\": \"chart_8.png\", \"task\": \"<OCR>\"}\n\n← OUTPUT:\n\"[OCR Data extracted from image chart_8.png]\""
  },
  "security_audit": {
    name: "SLM Security Audit",
    category: "Data & Utilities",
    catClass: "badge-data",
    stage: "Verified (CPU)",
    desc: "Guardrail system that scans inputs and outputs for PII leaks, system command injections, and safety violations before model execution.",
    features: [
      "Offline regex and semantic PII filters (SSN, credit cards, emails)",
      "System command injection and prompt jailbreak scanners",
      "Output evaluator to block harmful, invalid, or off-topic outputs",
      "Extremely fast CPU footprint — checks query in under 5ms"
    ],
    code: "from slm_security import SLMSecurityAudit\n\nauditor = SLMSecurityAudit()\nsafe_query = auditor.sanitize(\"SSN is 000-11-2222\")\nprint(safe_query)",
    input_output: "→ INPUT:\n\"SSN is 000-11-2222\"\n\n← OUTPUT:\n{\n  'safe': True,\n  'sanitized_text': 'SSN is [REDACTED_SSN]'\n}"
  },
  "embeddings_server": {
    name: "SLM Embeddings Server",
    category: "Data & Utilities",
    catClass: "badge-data",
    stage: "Verified (CPU)",
    desc: "Starts a local CPU-optimized embedding server to compute dense document and query vectors on standard hardware.",
    features: [
      "Loads quantized mini-LM or BGE embeddings locally",
      "High-speed cosine similarity index built directly in memory",
      "Provides local HTTP API endpoint for integration",
      "Under 200 MB RAM memory usage footprint during idle states"
    ],
    code: "from slm_embeddings import SLMEmbeddingsServer\n\nserver = SLMEmbeddingsServer()\nvector = server.embed([\"sample test\"])\nprint(vector)",
    input_output: "→ INPUT:\n\"sample test\"\n\n← OUTPUT:\n\"Vector dimension check: 1024\""
  }
};

// Sidebar Toggle Function
function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  if (sidebar) {
    sidebar.classList.toggle('open');
  }
}

// Open Upcoming Agent Modal
function openAgentModal(key) {
  const agent = UPCOMING_AGENTS[key];
  if (!agent) return;
  
  const modal = document.getElementById('agent-modal');
  const title = document.getElementById('modal-title');
  const body = document.getElementById('modal-body');
  
  if (!modal || !title || !body) return;
  
  title.innerText = agent.name;
  
  // Format features list
  let featuresHtml = "<ul>";
  agent.features.forEach(f => {
    featuresHtml += `<li>${f}</li>`;
  });
  featuresHtml += "</ul>";
  
  body.innerHTML = `
    <div class="framework-meta" style="margin-bottom: 1rem;">
      <span class="agent-cat-tag ${agent.catClass}">${agent.category}</span>
      <span class="badge-soon" style="margin-bottom: 0; background: #059669; border-color: #059669; color: #fff;">${agent.stage}</span>
    </div>
    <p style="color: var(--text-secondary); line-height: 1.6; margin-bottom: 1.5rem;">${agent.desc}</p>
    
    <h4 style="color: var(--primary); font-size: 1rem; margin-bottom: 0.5rem;">Capabilities:</h4>
    <div style="margin-bottom: 1.5rem;">${featuresHtml}</div>
    
    <h4 style="color: var(--primary); font-size: 1rem; margin-bottom: 0.5rem;">API Usage:</h4>
    <div class="code-panel" style="margin-bottom: 1.5rem;">
      <div class="code-header">
        <div class="code-dots">
          <div class="code-dot"></div><div class="code-dot"></div><div class="code-dot"></div>
        </div>
        <div class="code-title">Python Code</div>
      </div>
      <pre><code>${agent.code}</code></pre>
    </div>
    
    <h4 style="color: var(--primary); font-size: 1rem; margin-bottom: 0.5rem;">Verified Input &amp; Output Log:</h4>
    <div class="code-panel" style="background: rgba(0,0,0,0.35); border-color: rgba(255,255,255,0.08);">
      <div class="code-header">
        <div class="code-dots">
          <div class="code-dot"></div><div class="code-dot"></div><div class="code-dot"></div>
        </div>
        <div class="code-title">Execution Console</div>
      </div>
      <pre><code style="color: #38bdf8; font-family: monospace;">${agent.input_output}</code></pre>
    </div>
  `;
  
  modal.classList.add('open');
  
  // Close sidebar on mobile when click happens
  const sidebar = document.getElementById('sidebar');
  if (sidebar) sidebar.classList.remove('open');
}

function closeAgentModal() {
  const modal = document.getElementById('agent-modal');
  if (modal) {
    modal.classList.remove('open');
  }
}

// Category filtering and Search inside the main index page (if elements exist)
let activeCategory = 'all';

function filterCategory(btn, category) {
  const tabContainer = btn.parentElement;
  const buttons = tabContainer.querySelectorAll('.category-tab-btn');
  buttons.forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  
  activeCategory = category;
  applyFilters();
}

function applyFilters() {
  const searchInput = document.getElementById('agent-search');
  if (!searchInput) return;
  
  const searchQuery = searchInput.value.toLowerCase();
  const cards = document.querySelectorAll('.upcoming-card');
  
  cards.forEach(card => {
    const title = card.querySelector('h3').innerText.toLowerCase();
    const description = card.querySelector('p').innerText.toLowerCase();
    const cardCategory = card.getAttribute('data-category');
    
    const matchesSearch = title.includes(searchQuery) || description.includes(searchQuery);
    const matchesCategory = activeCategory === 'all' || cardCategory === activeCategory;
    
    if (matchesSearch && matchesCategory) {
      card.style.display = 'block';
    } else {
      card.style.display = 'none';
    }
  });
}

// Populate Sidebar lists dynamically on page load
document.addEventListener("DOMContentLoaded", () => {
  // Populate Active Libraries in sidebar
  const activeList = document.getElementById("sidebar-active-list");
  if (activeList) {
    activeList.innerHTML = `
      <li class="sidebar-item" id="nav-home"><a href="index.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg> Home</a></li>
      
      <!-- Developer Tools Category -->
      <div class="sidebar-group-title" style="margin-top:1.2rem; font-size:0.7rem; color:#8b5cf6; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Developer Tools</div>
      <li class="sidebar-item" id="nav-orchestrator"><a href="orchestrator.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><line x1="22" y1="12" x2="18" y2="12"></line><line x1="6" y1="12" x2="2" y2="12"></line><line x1="12" y1="6" x2="12" y2="2"></line><line x1="12" y1="22" x2="12" y2="18"></line></svg> SLM Orchestrator</a></li>
      <li class="sidebar-item" id="nav-sql"><a href="sql.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg> SLM Text-to-SQL</a></li>
      <li class="sidebar-item" id="nav-code-interpreter"><a href="code_interpreter.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg> SLM Code Interpreter</a></li>
      <li class="sidebar-item" id="nav-git-copilot"><a href="git_copilot.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><circle cx="18" cy="18" r="3"></circle><circle cx="6" cy="6" r="3"></circle><circle cx="6" cy="18" r="3"></circle><path d="M18 15V9a4 4 0 0 0-4-4H9"></path><line x1="6" y1="9" x2="6" y2="15"></line></svg> SLM Git Co-pilot</a></li>
      <li class="sidebar-item" id="nav-database-migrator"><a href="database_migrator.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"></path></svg> SLM Database Migrator</a></li>
      
      <!-- Web & Scraping Category -->
      <div class="sidebar-group-title" style="margin-top:1.2rem; font-size:0.7rem; color:#06b6d4; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Web & Scraping</div>
      <li class="sidebar-item" id="nav-web-agent"><a href="web_agent.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><rect x="2" y="3" width="20" height="18" rx="2" ry="2"></rect><line x1="2" y1="8" x2="22" y2="8"></line><line x1="6" y1="6" x2="6" y2="6"></line><line x1="10" y1="6" x2="10" y2="6"></line></svg> SLM Web Agent</a></li>
      <li class="sidebar-item" id="nav-web-scraper"><a href="web_scraper.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon></svg> SLM Web Scraper</a></li>
      <li class="sidebar-item" id="nav-search-orchestrator"><a href="search_orchestrator.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg> SLM Search Orchestrator</a></li>
      
      <!-- Productivity Category -->
      <div class="sidebar-group-title" style="margin-top:1.2rem; font-size:0.7rem; color:#f59e0b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Productivity</div>
      <li class="sidebar-item" id="nav-summarizer"><a href="summarizer.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg> SLM Summarizer</a></li>
      <li class="sidebar-item" id="nav-rag"><a href="rag.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><path d="M12 22c5.523 0 10-2.239 10-5V5c0-2.761-4.477-5-10-5S2 2.239 2 5v12c0 2.761 4.477 5 10 5z"></path><path d="M2 5c0 2.761 4.477 5 10 5s10-2.239 10-5"></path><path d="M2 11c0 2.761 4.477 5 10 5s10-2.239 10-5"></path></svg> SLM RAG</a></li>
      <li class="sidebar-item" id="nav-cli"><a href="cli.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><polyline points="4 17 10 11 4 5"></polyline><line x1="12" y1="19" x2="20" y2="19"></line></svg> SLM CLI Agent</a></li>
      <li class="sidebar-item" id="nav-email-assistant"><a href="email_assistant.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path><polyline points="22,6 12,13 2,6"></polyline></svg> SLM Email Assistant</a></li>
      <li class="sidebar-item" id="nav-meeting-summarizer"><a href="meeting_summarizer.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg> SLM Meeting Summarizer</a></li>
      <li class="sidebar-item" id="nav-voice-agent"><a href="voice_agent.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg> SLM Voice Agent</a></li>
      <li class="sidebar-item" id="nav-memory-manager"><a href="memory_manager.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"></circle><circle cx="6" cy="6" r="3"></circle><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="6" r="3"></circle><circle cx="18" cy="18" r="3"></circle><line x1="6" y1="9" x2="9" y2="12"></line><line x1="6" y1="15" x2="9" y2="12"></line><line x1="18" y1="9" x2="15" y2="12"></line><line x1="18" y1="15" x2="15" y2="12"></line></svg> SLM Memory Manager</a></li>
      <li class="sidebar-item" id="nav-task-planner"><a href="task_planner.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg> SLM Task Planner</a></li>
      <li class="sidebar-item" id="nav-pdf-chat"><a href="pdf_chat.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg> SLM PDF Chat</a></li>
      <li class="sidebar-item" id="nav-pkb-agent"><a href="pkb_agent.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><line x1="6" y1="3" x2="6" y2="15"></line><circle cx="18" cy="6" r="3"></circle><circle cx="6" cy="18" r="3"></circle><path d="M18 9a9 9 0 0 1-9 9"></path></svg> SLM PKB Agent</a></li>
      
      <!-- Data & Utilities Category -->
      <div class="sidebar-group-title" style="margin-top:1.2rem; font-size:0.7rem; color:#10b981; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Data & Utilities</div>
      <li class="sidebar-item" id="nav-json-cleaner"><a href="json_cleaner.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"></path></svg> SLM JSON Cleaner</a></li>
      <li class="sidebar-item" id="nav-document-parser"><a href="document_parser.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg> SLM Document Parser</a></li>
      <li class="sidebar-item" id="nav-vision-parser"><a href="vision_parser.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg> SLM Vision Parser</a></li>
      <li class="sidebar-item" id="nav-data-analyst"><a href="data_analyst.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg> SLM Data Analyst</a></li>
      <li class="sidebar-item" id="nav-translation-hub"><a href="translation_hub.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg> SLM Translation Hub</a></li>
      <li class="sidebar-item" id="nav-math-agent"><a href="math_agent.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><line x1="19" y1="5" x2="5" y2="19"></line><circle cx="6.5" cy="6.5" r="2.5"></circle><circle cx="17.5" cy="17.5" r="2.5"></circle></svg> SLM Math Agent</a></li>
      <li class="sidebar-item" id="nav-security-audit"><a href="security_audit.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg> SLM Security Audit</a></li>
      <li class="sidebar-item" id="nav-embeddings-server"><a href="embeddings_server.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="15" x2="23" y2="15"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="15" x2="4" y2="15"></line></svg> SLM Embeddings Server</a></li>
    `;
    
    // Highlight currently active page
    const path = window.location.pathname;
    const page = path.split("/").pop();
    if (page === "index.html" || page === "") {
      document.getElementById("nav-home")?.classList.add("active");
    } else if (page === "orchestrator.html") {
      document.getElementById("nav-orchestrator")?.classList.add("active");
    } else if (page === "rag.html") {
      document.getElementById("nav-rag")?.classList.add("active");
    } else if (page === "summarizer.html") {
      document.getElementById("nav-summarizer")?.classList.add("active");
    } else if (page === "sql.html") {
      document.getElementById("nav-sql")?.classList.add("active");
    } else if (page === "cli.html") {
      document.getElementById("nav-cli")?.classList.add("active");
    } else if (page === "code_interpreter.html") {
      document.getElementById("nav-code-interpreter")?.classList.add("active");
    } else if (page === "git_copilot.html") {
      document.getElementById("nav-git-copilot")?.classList.add("active");
    } else if (page === "json_cleaner.html") {
      document.getElementById("nav-json-cleaner")?.classList.add("active");
    } else if (page === "document_parser.html") {
      document.getElementById("nav-document-parser")?.classList.add("active");
    } else if (page === "vision_parser.html") {
      document.getElementById("nav-vision-parser")?.classList.add("active");
    } else if (page === "web_agent.html") {
      document.getElementById("nav-web-agent")?.classList.add("active");
    } else if (page === "web_scraper.html") {
      document.getElementById("nav-web-scraper")?.classList.add("active");
    } else if (page === "search_orchestrator.html") {
      document.getElementById("nav-search-orchestrator")?.classList.add("active");
    } else if (page === "database_migrator.html") {
      document.getElementById("nav-database-migrator")?.classList.add("active");
    } else if (page === "email_assistant.html") {
      document.getElementById("nav-email-assistant")?.classList.add("active");
    } else if (page === "meeting_summarizer.html") {
      document.getElementById("nav-meeting-summarizer")?.classList.add("active");
    } else if (page === "voice_agent.html") {
      document.getElementById("nav-voice-agent")?.classList.add("active");
    } else if (page === "memory_manager.html") {
      document.getElementById("nav-memory-manager")?.classList.add("active");
    } else if (page === "task_planner.html") {
      document.getElementById("nav-task-planner")?.classList.add("active");
    } else if (page === "pdf_chat.html") {
      document.getElementById("nav-pdf-chat")?.classList.add("active");
    } else if (page === "pkb_agent.html") {
      document.getElementById("nav-pkb-agent")?.classList.add("active");
    } else if (page === "data_analyst.html") {
      document.getElementById("nav-data-analyst")?.classList.add("active");
    } else if (page === "translation_hub.html") {
      document.getElementById("nav-translation-hub")?.classList.add("active");
    } else if (page === "math_agent.html") {
      document.getElementById("nav-math-agent")?.classList.add("active");
    } else if (page === "security_audit.html") {
      document.getElementById("nav-security-audit")?.classList.add("active");
    } else if (page === "embeddings_server.html") {
      document.getElementById("nav-embeddings-server")?.classList.add("active");
    }
  }

  // Remove the "Upcoming Ecosystem" sidebar group dynamically
  const upcomingList = document.getElementById("sidebar-upcoming-list");
  if (upcomingList) {
    const parentGroup = upcomingList.closest(".sidebar-group");
    if (parentGroup) {
      parentGroup.remove();
    }
  }

  // Close modal when clicking outside of modal content
  const modal = document.getElementById('agent-modal');
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        closeAgentModal();
      }
    });
  }
});
