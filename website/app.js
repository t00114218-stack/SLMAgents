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
    stage: "Planning Phase",
    desc: "Analyzes legacy database schemas and generates zero-downtime, CPU-optimized migrations and modern ORM model definitions offline.",
    features: [
      "Direct SQL table schema analysis and dependency mapping",
      "Automatic compatibility matching for migrations",
      "Generates modern SQLAlchemy and Django ORM models",
      "Suggests structural indexing plans for performance improvement"
    ],
    code: "from slm_db_migration import SLMDBMigrator\n\nmigrator = SLMDBMigrator()\nmigration_sql = migrator.generate_migration(from_schema, to_schema)\nprint(migration_sql)"
  },
  "email_assistant": {
    name: "SLM Email Assistant",
    category: "Productivity",
    catClass: "badge-prod",
    stage: "In Development",
    desc: "Securely processes your incoming inbox streams. Auto-drafts contexts, filters spam, and extracts urgent action items on standard CPUs.",
    features: [
      "Offline spam classifier and classification tagging",
      "Action item extraction and scheduled task planning",
      "Generates contextual email replies matching your custom tone profile",
      "PII protection — zero emails ever leave your machine"
    ],
    code: "from slm_email import SLMEmailAssistant\n\nassistant = SLMEmailAssistant()\nreply = assistant.draft_reply(email_text, context_brief)\nprint(reply)"
  },
  "meeting_summarizer": {
    name: "SLM Meeting Summarizer",
    category: "Productivity",
    catClass: "badge-prod",
    stage: "Planning Phase",
    desc: "Offline transcription post-processor. Distills meeting transcripts into action trackers, schedules, and bulleted logs with strict formatting rules.",
    features: [
      "Turns conversational text blocks into formal action tables",
      "Identifies speaker intent, decisions, and deadlines",
      "Map-Reduce pipeline support for 2-hour long transcription logs",
      "Strict template outputs matching markdown specifications"
    ],
    code: "from slm_meeting import SLMMeetingSummarizer\n\nsummarizer = SLMMeetingSummarizer()\ntodos = summarizer.extract_todos(transcript_text)\nprint(todos)"
  },
  "voice_agent": {
    name: "SLM Voice Agent",
    category: "Productivity",
    catClass: "badge-prod",
    stage: "Planning Phase",
    desc: "Fast offline conversational companion combining Whisper speech-to-text, 1.5B chat, and lightweight text-to-speech pipelines on CPU.",
    features: [
      "Offline audio-to-text speech transcription",
      "Low-latency response generation using quantized ONNX",
      "Text-to-speech synthesis utilizing local CPU synthesizer models",
      "Hands-free voice trigger support"
    ],
    code: "from slm_voice import SLMVoiceAgent\n\nvoice = SLMVoiceAgent()\nvoice.start_listening_loop()"
  },
  "memory_manager": {
    name: "SLM Memory Manager",
    category: "Productivity",
    catClass: "badge-prod",
    stage: "In Development",
    desc: "Manages long-term personal state and preference graphs. Learns and adapts to user query patterns locally without cloud synchronization.",
    features: [
      "Entities and relations extraction from chat history",
      "Builds a local knowledge graph of user preferences",
      "Prunes older irrelevant details to fit within context limits",
      "Auto-injects user context tags into RAG sessions"
    ],
    code: "from slm_memory import SLMMemoryManager\n\nmem = SLMMemoryManager()\nmem.store_fact(\"User lives in Seattle and prefers python code.\")\nprint(mem.get_relevant_facts(\"Where to launch app?\"))"
  },
  "task_planner": {
    name: "SLM Task Planner",
    category: "Productivity",
    catClass: "badge-prod",
    stage: "Planning Phase",
    desc: "Autonomous goal decomposition system. Breaks complex tasks into prioritized action items and assigns them to specialized local sub-agents.",
    features: [
      "Goal decomposition and sub-task scheduling",
      "Dependency mapping for parallel execution branches",
      "Runtime execution tracker with dynamic adjustment",
      "Fallback handler to revise tasks if a sub-agent fails"
    ],
    code: "from slm_task_planner import SLMTaskPlanner\n\nplanner = SLMTaskPlanner()\nplan = planner.build_plan(\"Build website and deploy to local host\")\nprint(plan)"
  },
  "pdf_chat": {
    name: "SLM PDF Chat",
    category: "Productivity",
    catClass: "badge-prod",
    stage: "In Development",
    desc: "Securely parses complex PDF documents. Assembles layouts, reads tables, and lets you chat with local legal contracts, research articles, or receipts.",
    features: [
      "Locally extracts layout text and multi-column paragraphs",
      "Parses database tables inside PDFs directly to list-of-dicts",
      "Built-in RAG chunk generator for offline querying",
      "Supports scanned image PDFs via local OCR integration"
    ],
    code: "from slm_pdf import SLMPDFChat\n\npdf = SLMPDFChat()\npdf.load(\"invoice.pdf\")\nans = pdf.ask(\"What is the total due amount?\")\nprint(ans)"
  },
  "pkb_agent": {
    name: "SLM PKB Agent",
    category: "Productivity",
    catClass: "badge-prod",
    stage: "Planning Phase",
    desc: "Local knowledge management assistant. Builds, links, and tags markdown documents in Obsidian, Notion, or Logseq vaults offline.",
    features: [
      "Auto-scans directories of markdown notes to map semantic clusters",
      "Suggests links between notes based on context similarity",
      "Auto-generates summaries, tags, and indexing logs for vault folders",
      "Integrates directly with local Obsidian vaults"
    ],
    code: "from slm_pkb import SLMPKBAgent\n\nagent = SLMPKBAgent(vault_path=\"~/Obsidian/MyVault\")\nagent.build_vault_graph()"
  },
  "data_analyst": {
    name: "SLM Data Analyst",
    category: "Data & Utilities",
    catClass: "badge-data",
    stage: "Planning Phase",
    desc: "Loads local CSV, Parquet, or Excel files. Answers statistical questions, performs calculations, and auto-generates data visualization code.",
    features: [
      "Direct pandas dataframe parsing and stats calculator",
      "Translates user query into python matplotlib/pandas code blocks",
      "Generates summary tables and column distribution charts",
      "100% offline analysis of highly sensitive company sheets"
    ],
    code: "from slm_data import SLMDataAnalyst\n\nanalyst = SLMDataAnalyst()\nscript = analyst.generate_analysis(\"sales.csv\", \"plot revenue by department\")\nprint(script)"
  },
  "translation_hub": {
    name: "SLM Translation Hub",
    category: "Data & Utilities",
    catClass: "badge-data",
    stage: "In Development",
    desc: "Quantized multilingual translation library designed for offline local document conversion across 20+ language profiles.",
    features: [
      "Quantized translation weights optimized for CPU RAM footprint",
      "Preserves original formatting (HTML, Markdown, DOCX markup)",
      "Sentence-alignment validation for precise paragraph mappings",
      "Completely offline operation — ideal for restricted documents"
    ],
    code: "from slm_translation import SLMTranslationHub\n\nhub = SLMTranslationHub()\ntranslated = hub.translate(\"Hello World\", target_lang=\"de\")\nprint(translated)"
  },
  "math_agent": {
    name: "SLM Math Agent",
    category: "Data & Utilities",
    catClass: "badge-data",
    stage: "Planning Phase",
    desc: "Specialized arithmetic reasoning model. Handles math formulations, algebraic simplifications, and steps through complex equations offline.",
    features: [
      "Symbolic algebra calculator mapping using local SymPy",
      "Parses equations and graphs steps to final result",
      "Verifies intermediate steps to prevent math hallucinations",
      "Optimized math tokens prompt training templates"
    ],
    code: "from slm_math import SLMMathAgent\n\nagent = SLMMathAgent()\nsteps = agent.solve(\"integrate x^2 from 0 to 3\")\nprint(steps)"
  },
  "vision_parser": {
    name: "SLM Vision Parser",
    category: "Data & Utilities",
    catClass: "badge-data",
    stage: "Planning Phase",
    desc: "Offline chart, diagram, and whiteboard reader. Converts scanned infographics and drawings to clean structured text summaries.",
    features: [
      "Quantized local Vision-Language model (VLM) weights",
      "Extracts key numbers and trends from bar, line, and pie charts",
      "OCR reader for whiteboards and handwritten flowcharts",
      "Translates infographics directly to clean markdown tables"
    ],
    code: "from slm_vision import SLMVisionParser\n\nparser = SLMVisionParser()\nchart_info = parser.parse_chart(\"sales_chart.png\")\nprint(chart_info)"
  },
  "security_audit": {
    name: "SLM Security Audit",
    category: "Data & Utilities",
    catClass: "badge-data",
    stage: "In Development",
    desc: "Guardrail system that scans inputs and outputs for PII leaks, system command injections, and safety violations before model execution.",
    features: [
      "Offline regex and semantic PII filters (SSN, credit cards, emails)",
      "System command injection and prompt jailbreak scanners",
      "Output evaluator to block harmful, invalid, or off-topic outputs",
      "Extremely fast CPU footprint — checks query in under 5ms"
    ],
    code: "from slm_security import SLMSecurityAudit\n\nauditor = SLMSecurityAudit()\nsafe_query = auditor.sanitize(\"My email is test@email.com, query database\")\nprint(safe_query)"
  },
  "embeddings_server": {
    name: "SLM Embeddings Server",
    category: "Data & Utilities",
    catClass: "badge-data",
    stage: "In Development",
    desc: "Starts a local CPU-optimized embedding server to compute dense document and query vectors on standard hardware.",
    features: [
      "Loads quantized mini-LM or BGE embeddings locally",
      "High-speed cosine similarity index built directly in memory",
      "Provides local HTTP API endpoint for integration",
      "Under 200 MB RAM memory usage footprint during idle states"
    ],
    code: "from slm_embeddings import SLMEmbeddingsServer\n\nserver = SLMEmbeddingsServer()\nvector = server.embed(\"Text to vectorize\")\nprint(vector[:5])"
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
      <span class="badge-soon" style="margin-bottom: 0;">${agent.stage}</span>
    </div>
    <p style="color: var(--text-secondary); line-height: 1.6; margin-bottom: 1.5rem;">${agent.desc}</p>
    
    <h4 style="color: var(--primary); font-size: 1rem; margin-bottom: 0.5rem;">Planned Capabilities:</h4>
    <div style="margin-bottom: 1.5rem;">${featuresHtml}</div>
    
    <h4 style="color: var(--primary); font-size: 1rem; margin-bottom: 0.5rem;">Planned API Usage:</h4>
    <div class="code-panel">
      <div class="code-header">
        <div class="code-dots">
          <div class="code-dot"></div><div class="code-dot"></div><div class="code-dot"></div>
        </div>
        <div class="code-title">Python Code</div>
      </div>
      <pre><code>${agent.code}</code></pre>
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
      <li class="sidebar-item" id="nav-orchestrator"><a href="orchestrator.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><line x1="22" y1="12" x2="18" y2="12"></line><line x1="6" y1="12" x2="2" y2="12"></line><line x1="12" y1="6" x2="12" y2="2"></line><line x1="12" y1="22" x2="12" y2="18"></line></svg> SLM Orchestrator</a></li>
      <li class="sidebar-item" id="nav-rag"><a href="rag.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><path d="M12 22c5.523 0 10-2.239 10-5V5c0-2.761-4.477-5-10-5S2 2.239 2 5v12c0 2.761 4.477 5 10 5z"></path><path d="M2 5c0 2.761 4.477 5 10 5s10-2.239 10-5"></path><path d="M2 11c0 2.761 4.477 5 10 5s10-2.239 10-5"></path></svg> SLM RAG</a></li>
      <li class="sidebar-item" id="nav-summarizer"><a href="summarizer.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg> SLM Summarizer</a></li>
      <li class="sidebar-item" id="nav-sql"><a href="sql.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg> SLM Text-to-SQL</a></li>
      <li class="sidebar-item" id="nav-cli"><a href="cli.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><polyline points="4 17 10 11 4 5"></polyline><line x1="12" y1="19" x2="20" y2="19"></line></svg> SLM CLI Agent</a></li>
      <li class="sidebar-item" id="nav-code-interpreter"><a href="code_interpreter.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg> SLM Code Interpreter</a></li>
      <li class="sidebar-item" id="nav-git-copilot"><a href="git_copilot.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><circle cx="18" cy="18" r="3"></circle><circle cx="6" cy="6" r="3"></circle><circle cx="6" cy="18" r="3"></circle><path d="M18 15V9a4 4 0 0 0-4-4H9"></path><line x1="6" y1="9" x2="6" y2="15"></line></svg> SLM Git Co-pilot</a></li>
      <li class="sidebar-item" id="nav-json-cleaner"><a href="json_cleaner.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"></path></svg> SLM JSON Cleaner</a></li>
      <li class="sidebar-item" id="nav-document-parser"><a href="document_parser.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg> SLM Document Parser</a></li>
      <li class="sidebar-item" id="nav-vision-parser"><a href="vision_parser.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg> SLM Vision Parser</a></li>
      <li class="sidebar-item" id="nav-web-agent"><a href="web_agent.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><rect x="2" y="3" width="20" height="18" rx="2" ry="2"></rect><line x1="2" y1="8" x2="22" y2="8"></line><line x1="6" y1="6" x2="6" y2="6"></line><line x1="10" y1="6" x2="10" y2="6"></line></svg> SLM Web Agent</a></li>
      <li class="sidebar-item" id="nav-web-scraper"><a href="web_scraper.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon></svg> SLM Web Scraper</a></li>
      <li class="sidebar-item" id="nav-search-orchestrator"><a href="search_orchestrator.html"><svg class="sidebar-icon" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg> SLM Search Orchestrator</a></li>
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
    }
  }

  // Populate Upcoming Agents categorized lists in sidebar
  const upcomingList = document.getElementById("sidebar-upcoming-list");
  if (upcomingList) {
    let listHtml = "";
    
    // Developer Tools
    listHtml += `<div class="sidebar-group-title" style="margin-top:1rem; font-size:0.7rem; color:#8b5cf6;">Developer Tools</div>`;
    Object.keys(UPCOMING_AGENTS).forEach(key => {
      const a = UPCOMING_AGENTS[key];
      if (a.category === "Developer Tools") {
        listHtml += `<li class="sidebar-item"><button onclick="openAgentModal('${key}')"><span class="sidebar-bullet"></span> ${a.name}</button></li>`;
      }
    });

    // Web & Scraping
    listHtml += `<div class="sidebar-group-title" style="margin-top:1rem; font-size:0.7rem; color:#06b6d4;">Web & Scraping</div>`;
    Object.keys(UPCOMING_AGENTS).forEach(key => {
      const a = UPCOMING_AGENTS[key];
      if (a.category === "Web & Scraping") {
        listHtml += `<li class="sidebar-item"><button onclick="openAgentModal('${key}')"><span class="sidebar-bullet"></span> ${a.name}</button></li>`;
      }
    });

    // Productivity
    listHtml += `<div class="sidebar-group-title" style="margin-top:1rem; font-size:0.7rem; color:#f59e0b;">Productivity</div>`;
    Object.keys(UPCOMING_AGENTS).forEach(key => {
      const a = UPCOMING_AGENTS[key];
      if (a.category === "Productivity") {
        listHtml += `<li class="sidebar-item"><button onclick="openAgentModal('${key}')"><span class="sidebar-bullet"></span> ${a.name}</button></li>`;
      }
    });

    // Data & Utilities
    listHtml += `<div class="sidebar-group-title" style="margin-top:1rem; font-size:0.7rem; color:#10b981;">Data & Utilities</div>`;
    Object.keys(UPCOMING_AGENTS).forEach(key => {
      const a = UPCOMING_AGENTS[key];
      if (a.category === "Data & Utilities") {
        listHtml += `<li class="sidebar-item"><button onclick="openAgentModal('${key}')"><span class="sidebar-bullet"></span> ${a.name}</button></li>`;
      }
    });

    upcomingList.innerHTML = listHtml;
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
