package com.slmagents.offline.agents

enum class AgentType(val id: String, val displayName: String, val iconName: String, val shortDescription: String) {
    ORCHESTRATOR("auto_orchestrator", "Auto Orchestrator", "hub", "Intelligently routes prompts to the best specialized agent"),
    GENERAL("general", "General Assistant", "chat", "Versatile offline conversational SLM for questions and dialogue"),
    RAG("rag_agent", "Document RAG Agent", "menu_book", "Retrieves context from imported offline documents and notes"),
    SUMMARIZER("summarizer", "Summarizer Agent", "summarize", "Condenses long text, transcripts, and articles into clean briefs"),
    TASK_PLANNER("task_planner", "Task Planner", "checklist", "Breaks complex objectives into actionable step-by-step plans"),
    LOCAL_TOOLS("local_tools", "Local Tools Agent", "build", "Executes offline calculations, data filtering, and formatting")
}

data class AgentConfig(
    val type: AgentType,
    val name: String = type.displayName,
    val systemPrompt: String,
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val maxTokens: Int = 1024
)
