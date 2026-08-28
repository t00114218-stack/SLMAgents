package com.slmagents.offline.agents

class SummarizerAgent {

    val config = AgentConfig(
        type = AgentType.SUMMARIZER,
        systemPrompt = """
            You are an expert Offline Summarizer Agent.
            Your goal is to extract core insights from long texts, meeting notes, transcripts, and documents.
            Structure your output with:
            1. **TL;DR** (1-2 sentences)
            2. **Key Takeaways** (concise bullet points)
            3. **Action Items / Next Steps** (if applicable)
            Be concise, clear, and omit unnecessary filler.
        """.trimIndent(),
        temperature = 0.4f,
        topP = 0.9f,
        maxTokens = 1200
    )
}
