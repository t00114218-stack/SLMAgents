package com.slmagents.offline.agents

import com.slmagents.offline.core.InferenceConfig
import com.slmagents.offline.core.LlamaEngine
import com.slmagents.offline.db.ChatDao
import com.slmagents.offline.db.ChatMessageEntity
import com.slmagents.offline.db.ConversationEntity
import com.slmagents.offline.db.VectorStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.util.UUID

class Orchestrator(
    private val llamaEngine: LlamaEngine,
    private val chatDao: ChatDao,
    private val vectorStore: VectorStore
) {
    private val router = AgentRouter()
    val ragAgent = RAGAgent(vectorStore)
    val summarizerAgent = SummarizerAgent()
    val plannerAgent = TaskPlannerAgent()
    val toolAgent = ToolAgent()

    val generalConfig = AgentConfig(
        type = AgentType.GENERAL,
        systemPrompt = "You are a helpful, concise, and capable offline AI assistant running locally on an Android device.",
        temperature = 0.7f,
        topP = 0.9f,
        maxTokens = 1024
    )

    fun getAgentConfig(agentType: AgentType): AgentConfig {
        return when (agentType) {
            AgentType.ORCHESTRATOR -> generalConfig.copy(name = "Auto Orchestrator")
            AgentType.GENERAL -> generalConfig
            AgentType.RAG -> ragAgent.config
            AgentType.SUMMARIZER -> summarizerAgent.config
            AgentType.TASK_PLANNER -> plannerAgent.config
            AgentType.LOCAL_TOOLS -> toolAgent.config
        }
    }

    suspend fun createConversation(title: String, activeAgent: AgentType = AgentType.ORCHESTRATOR): ConversationEntity {
        val conv = ConversationEntity(
            id = UUID.randomUUID().toString(),
            title = title.ifBlank { "New Offline Chat" },
            createdAt = System.currentTimeMillis(),
            activeAgentId = activeAgent.id
        )
        chatDao.insertConversation(conv)
        return conv
    }

    suspend fun sendMessage(
        conversationId: String,
        userPrompt: String,
        selectedAgentType: AgentType
    ): Flow<Pair<String, AgentType>> = flow {
        // 1. Record user message in DB
        val userMsgId = UUID.randomUUID().toString()
        chatDao.insertMessage(
            ChatMessageEntity(
                id = userMsgId,
                conversationId = conversationId,
                role = "user",
                content = userPrompt,
                agentId = selectedAgentType.id,
                timestamp = System.currentTimeMillis()
            )
        )

        // 2. Determine target agent
        val targetAgent = if (selectedAgentType == AgentType.ORCHESTRATOR) {
            val docs = chatDao.getAllChunks()
            router.route(userPrompt, hasDocuments = docs.isNotEmpty())
        } else {
            selectedAgentType
        }

        // 3. Resolve context and prompt
        val effectivePrompt = when (targetAgent) {
            AgentType.RAG -> ragAgent.formatPromptWithContext(userPrompt)
            else -> userPrompt
        }

        val config = getAgentConfig(targetAgent)

        // 4. Fetch recent chat history
        val pastMessages = chatDao.getMessagesList(conversationId)
            .takeLast(6)
            .map { it.role to it.content }

        val formattedPrompt = LlamaEngine.formatChatML(
            systemPrompt = config.systemPrompt,
            history = pastMessages,
            userMessage = effectivePrompt
        )

        // 5. Stream inference
        val assistantResponseBuilder = StringBuilder()
        val streamConfig = InferenceConfig(
            temperature = config.temperature,
            topP = config.topP,
            maxTokens = config.maxTokens
        )

        llamaEngine.generateStream(formattedPrompt, streamConfig).collect { token ->
            assistantResponseBuilder.append(token)
            emit(token to targetAgent)
        }

        // 6. Save assistant response to Room
        val assistantMsgId = UUID.randomUUID().toString()
        val stats = llamaEngine.currentStats
        chatDao.insertMessage(
            ChatMessageEntity(
                id = assistantMsgId,
                conversationId = conversationId,
                role = "assistant",
                content = assistantResponseBuilder.toString(),
                agentId = targetAgent.id,
                tokensPerSec = stats.tokensPerSecond,
                timestamp = System.currentTimeMillis()
            )
        )
    }
}
