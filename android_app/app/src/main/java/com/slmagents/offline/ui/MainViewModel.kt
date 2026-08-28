package com.slmagents.offline.ui

import android.app.Application
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.slmagents.offline.agents.AgentType
import com.slmagents.offline.agents.Orchestrator
import com.slmagents.offline.core.LlamaEngine
import com.slmagents.offline.core.ModelInfo
import com.slmagents.offline.core.ModelManager
import com.slmagents.offline.db.AppDatabase
import com.slmagents.offline.db.ChatMessageEntity
import com.slmagents.offline.db.ConversationEntity
import com.slmagents.offline.db.DocumentEntity
import com.slmagents.offline.db.VectorStore
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.io.File

data class UiState(
    val selectedAgent: AgentType = AgentType.ORCHESTRATOR,
    val isGenerating: Boolean = false,
    val streamingText: String = "",
    val streamingAgent: AgentType = AgentType.GENERAL,
    val currentTokensPerSec: Float = 0f,
    val memoryUsageMb: Long = 0,
    val activeConversationId: String? = null,
    val activeModelName: String = "No Model Loaded",
    val isModelLoaded: Boolean = false,
    val statusMessage: String? = null
)

class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val db = AppDatabase.getDatabase(application)
    private val chatDao = db.chatDao()
    val vectorStore = VectorStore(chatDao)
    val llamaEngine = LlamaEngine(application)
    val modelManager = ModelManager(application)
    val orchestrator = Orchestrator(llamaEngine, chatDao, vectorStore)

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    val conversations: StateFlow<List<ConversationEntity>> = chatDao.getAllConversations()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    val documents: StateFlow<List<DocumentEntity>> = chatDao.getAllDocuments()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    private val _messages = MutableStateFlow<List<ChatMessageEntity>>(emptyList())
    val messages: StateFlow<List<ChatMessageEntity>> = _messages.asStateFlow()

    private val _localModels = MutableStateFlow<List<ModelInfo>>(emptyList())
    val localModels: StateFlow<List<ModelInfo>> = _localModels.asStateFlow()

    init {
        refreshModels()
        initDefaultConversation()
    }

    fun refreshModels() {
        viewModelScope.launch {
            val models = modelManager.listLocalModels()
            _localModels.value = models
            if (models.isNotEmpty() && !llamaEngine.isModelLoaded) {
                loadModel(models.first())
            }
        }
    }

    fun loadModel(modelInfo: ModelInfo) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(statusMessage = "Loading model: ${modelInfo.name}...")
            val result = llamaEngine.loadModel(File(modelInfo.filePath))
            if (result.isSuccess) {
                _uiState.value = _uiState.value.copy(
                    activeModelName = modelInfo.name,
                    isModelLoaded = true,
                    statusMessage = "Model '${modelInfo.name}' loaded successfully",
                    memoryUsageMb = llamaEngine.getUsedMemoryMb()
                )
            } else {
                _uiState.value = _uiState.value.copy(
                    statusMessage = "Failed to load model: ${result.exceptionOrNull()?.message}"
                )
            }
        }
    }

    fun unloadModel() {
        viewModelScope.launch {
            llamaEngine.unloadModel()
            _uiState.value = _uiState.value.copy(
                activeModelName = "No Model Loaded",
                isModelLoaded = false,
                statusMessage = "Model unloaded",
                memoryUsageMb = 0
            )
        }
    }

    fun importModelUri(uri: Uri) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(statusMessage = "Importing GGUF model...")
            val result = modelManager.importModelFromUri(uri)
            if (result.isSuccess) {
                refreshModels()
                _uiState.value = _uiState.value.copy(statusMessage = "Model imported! Loading now...")
                result.getOrNull()?.let { file ->
                    val loadRes = llamaEngine.loadModel(file)
                    if (loadRes.isSuccess) {
                        _uiState.value = _uiState.value.copy(
                            activeModelName = file.nameWithoutExtension.replace("-", " ").replace("_", " "),
                            isModelLoaded = true,
                            statusMessage = "Model loaded and ready for offline AI chat!"
                        )
                    }
                }
            } else {
                _uiState.value = _uiState.value.copy(statusMessage = "Failed to import model: ${result.exceptionOrNull()?.message}")
            }
        }
    }

    fun deleteLocalModel(modelInfo: ModelInfo) {
        viewModelScope.launch {
            if (_uiState.value.activeModelName.contains(modelInfo.name, ignoreCase = true)) {
                unloadModel()
            }
            val deleted = modelManager.deleteModelFile(modelInfo.filePath)
            if (deleted) {
                _uiState.value = _uiState.value.copy(statusMessage = "Deleted model file")
                refreshModels()
            } else {
                _uiState.value = _uiState.value.copy(statusMessage = "Could not delete model file")
            }
        }
    }

    private fun initDefaultConversation() {
        viewModelScope.launch {
            conversations.collect { list ->
                if (list.isEmpty()) {
                    val newConv = orchestrator.createConversation("General Chat")
                    selectConversation(newConv.id)
                } else if (_uiState.value.activeConversationId == null) {
                    selectConversation(list.first().id)
                }
            }
        }
    }

    fun selectConversation(id: String) {
        _uiState.value = _uiState.value.copy(activeConversationId = id)
        viewModelScope.launch {
            chatDao.getMessagesForConversation(id).collect {
                _messages.value = it
            }
        }
    }

    fun createNewConversation(title: String) {
        viewModelScope.launch {
            val conv = orchestrator.createConversation(title, _uiState.value.selectedAgent)
            selectConversation(conv.id)
        }
    }

    fun selectAgent(agentType: AgentType) {
        _uiState.value = _uiState.value.copy(selectedAgent = agentType)
    }

    fun sendMessage(promptText: String) {
        val convId = _uiState.value.activeConversationId ?: return
        if (promptText.isBlank() || _uiState.value.isGenerating) return

        _uiState.value = _uiState.value.copy(
            isGenerating = true,
            streamingText = "",
            currentTokensPerSec = 0f
        )

        viewModelScope.launch {
            try {
                orchestrator.sendMessage(
                    conversationId = convId,
                    userPrompt = promptText,
                    selectedAgentType = _uiState.value.selectedAgent
                ).collect { (token, activeAgent) ->
                    _uiState.value = _uiState.value.copy(
                        streamingText = _uiState.value.streamingText + token,
                        streamingAgent = activeAgent
                    )
                }
            } catch (e: Exception) {
                Log.e("MainViewModel", "Inference error: ${e.message}", e)
            } finally {
                val stats = llamaEngine.currentStats
                _uiState.value = _uiState.value.copy(
                    isGenerating = false,
                    streamingText = "",
                    currentTokensPerSec = stats.tokensPerSecond,
                    memoryUsageMb = llamaEngine.getUsedMemoryMb()
                )
            }
        }
    }

    fun stopGeneration() {
        llamaEngine.stopGeneration()
        _uiState.value = _uiState.value.copy(isGenerating = false)
    }

    fun ingestDocument(title: String, filename: String, text: String) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(statusMessage = "Indexing document into local vector store...")
            val result = vectorStore.ingestDocument(title, filename, text)
            if (result.isSuccess) {
                _uiState.value = _uiState.value.copy(statusMessage = "Document indexed offline!")
            } else {
                _uiState.value = _uiState.value.copy(statusMessage = "Failed to index document: ${result.exceptionOrNull()?.message}")
            }
        }
    }

    fun deleteDocument(docId: String) {
        viewModelScope.launch {
            chatDao.deleteDocument(docId)
        }
    }

    fun clearStatus() {
        _uiState.value = _uiState.value.copy(statusMessage = null)
    }
}
