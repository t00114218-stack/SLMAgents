package com.slmagents.offline.agents

class AgentRouter {

    fun route(prompt: String, hasDocuments: Boolean): AgentType {
        val lower = prompt.lowercase().trim()

        // 1. Check for summarization keywords
        if (lower.startsWith("summarize") || lower.startsWith("summary") || lower.contains("tldr") || 
            lower.contains("brief overview") || lower.contains("key takeaways") || lower.contains("bullet points of")) {
            return AgentType.SUMMARIZER
        }

        // 2. Check for task planning / steps
        if (lower.startsWith("plan") || lower.startsWith("how to") || lower.contains("step by step") || 
            lower.contains("roadmap") || lower.contains("implementation plan") || lower.contains("break down the task")) {
            return AgentType.TASK_PLANNER
        }

        // 3. Check for math / calculation / tools
        if (lower.startsWith("calculate") || lower.startsWith("compute") || lower.matches(Regex(".*[0-9]+\\s*[*+/\\-^%]+\\s*[0-9]+.*"))) {
            return AgentType.LOCAL_TOOLS
        }

        // 4. Check for document queries if local documents exist
        if (hasDocuments && (lower.contains("in the document") || lower.contains("according to") || 
            lower.contains("from the file") || lower.contains("in my notes") || lower.contains("search for"))) {
            return AgentType.RAG
        }

        return AgentType.GENERAL
    }
}
