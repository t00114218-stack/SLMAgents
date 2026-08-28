package com.slmagents.offline.agents

import com.slmagents.offline.db.VectorStore

class RAGAgent(private val vectorStore: VectorStore) {

    val config = AgentConfig(
        type = AgentType.RAG,
        systemPrompt = """
            You are a specialized Offline Document RAG Agent.
            Your job is to answer user questions truthfully and accurately using ONLY the provided document context below.
            If the context does not contain the answer, state clearly that the information is not in the imported documents.
            Always cite facts directly from the context.
        """.trimIndent(),
        temperature = 0.3f, // Lower temperature for factual retrieval
        topP = 0.85f,
        maxTokens = 1024
    )

    suspend fun formatPromptWithContext(userQuery: String): String {
        val searchResults = vectorStore.searchRelevantChunks(userQuery, topK = 3)
        if (searchResults.isEmpty()) {
            return userQuery
        }

        val contextBuilder = StringBuilder()
        contextBuilder.append("--- RELEVANT DOCUMENT CONTEXT ---\n")
        searchResults.forEachIndexed { i, res ->
            contextBuilder.append("[Context Chunk ${i + 1}]:\n")
            contextBuilder.append(res.chunk.content.trim())
            contextBuilder.append("\n\n")
        }
        contextBuilder.append("--- END OF CONTEXT ---\n\n")
        contextBuilder.append("User Question: ").append(userQuery)

        return contextBuilder.toString()
    }
}
