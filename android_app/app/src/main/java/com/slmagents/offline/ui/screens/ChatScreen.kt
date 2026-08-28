package com.slmagents.offline.ui.screens

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.widget.Toast
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.slmagents.offline.agents.AgentType
import com.slmagents.offline.db.ChatMessageEntity
import com.slmagents.offline.ui.MainViewModel
import com.slmagents.offline.ui.theme.*
import kotlinx.coroutines.launch

@Composable
fun ChatScreen(viewModel: MainViewModel) {
    val uiState by viewModel.uiState.collectAsState()
    val messages by viewModel.messages.collectAsState()
    val context = LocalContext.current
    var inputText by remember { mutableStateOf("") }
    val listState = rememberLazyListState()
    val coroutineScope = rememberCoroutineScope()

    LaunchedEffect(messages.size, uiState.streamingText) {
        if (messages.isNotEmpty() || uiState.streamingText.isNotEmpty()) {
            listState.animateScrollToItem((messages.size + (if (uiState.isGenerating) 1 else 0)).coerceAtLeast(0))
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        // Top Bar: Performance Stats & Status
        Surface(
            color = MaterialTheme.colorScheme.surface,
            tonalElevation = 3.dp,
            modifier = Modifier.fillMaxWidth()
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .clip(CircleShape)
                                .background(if (uiState.isModelLoaded) AccentEmerald else AccentAmber)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = if (uiState.isModelLoaded) uiState.activeModelName else "No Offline Model",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                    Text(
                        text = "100% Offline Mode" + if (uiState.currentTokensPerSec > 0) " • %.1f tok/s".format(uiState.currentTokensPerSec) else "",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                IconButton(
                    onClick = { viewModel.createNewConversation("New Chat") }
                ) {
                    Icon(
                        Icons.Default.AddComment,
                        contentDescription = "New Chat",
                        tint = MaterialTheme.colorScheme.primary
                    )
                }
            }
        }

        // Agent Switcher Strip
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState())
                .padding(horizontal = 12.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            AgentType.values().forEach { agent ->
                val isSelected = uiState.selectedAgent == agent
                FilterChip(
                    selected = isSelected,
                    onClick = { viewModel.selectAgent(agent) },
                    label = {
                        Text(
                            agent.displayName,
                            fontSize = 12.sp,
                            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
                        )
                    },
                    leadingIcon = {
                        Icon(
                            imageVector = when (agent) {
                                AgentType.ORCHESTRATOR -> Icons.Default.SmartToy
                                AgentType.GENERAL -> Icons.Default.Chat
                                AgentType.RAG -> Icons.Default.MenuBook
                                AgentType.SUMMARIZER -> Icons.Default.Summarize
                                AgentType.TASK_PLANNER -> Icons.Default.Checklist
                                AgentType.LOCAL_TOOLS -> Icons.Default.Build
                            },
                            contentDescription = null,
                            modifier = Modifier.size(16.dp),
                            tint = if (isSelected) PrimaryCyan else MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    },
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = SurfaceLightDark,
                        selectedLabelColor = PrimaryCyan
                    )
                )
            }
        }

        Divider(color = MaterialTheme.colorScheme.surfaceVariant, thickness = 0.5.dp)

        // Chat Message List
        Box(modifier = Modifier.weight(1f)) {
            if (messages.isEmpty() && uiState.streamingText.isEmpty()) {
                EmptyChatSuggestions(
                    activeAgent = uiState.selectedAgent,
                    onSelectPrompt = { prompt ->
                        inputText = prompt
                        viewModel.sendMessage(prompt)
                        inputText = ""
                    }
                )
            } else {
                LazyColumn(
                    state = listState,
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(horizontal = 12.dp),
                    contentPadding = PaddingValues(vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    items(messages) { msg ->
                        MessageBubble(
                            message = msg,
                            onCopy = {
                                val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                                clipboard.setPrimaryClip(ClipData.newPlainText("SLM Chat", msg.content))
                                Toast.makeText(context, "Copied to clipboard", Toast.LENGTH_SHORT).show()
                            }
                        )
                    }

                    if (uiState.isGenerating && uiState.streamingText.isNotEmpty()) {
                        item {
                            StreamingBubble(
                                text = uiState.streamingText,
                                agent = uiState.streamingAgent
                            )
                        }
                    }
                }
            }
        }

        // Status banner if present
        AnimatedVisibility(visible = uiState.statusMessage != null) {
            uiState.statusMessage?.let { status ->
                Surface(
                    color = AccentIndigo.copy(alpha = 0.15f),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 4.dp),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(status, style = MaterialTheme.typography.bodySmall, color = PrimaryCyan)
                        IconButton(
                            onClick = { viewModel.clearStatus() },
                            modifier = Modifier.size(20.dp)
                        ) {
                            Icon(Icons.Default.Close, contentDescription = "Dismiss", tint = Color.Gray)
                        }
                    }
                }
            }
        }

        // Input Field & Action Bar
        Surface(
            color = MaterialTheme.colorScheme.surface,
            tonalElevation = 4.dp,
            modifier = Modifier.fillMaxWidth()
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = inputText,
                    onValueChange = { inputText = it },
                    placeholder = {
                        Text(
                            text = "Ask ${uiState.selectedAgent.displayName}...",
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            fontSize = 14.sp
                        )
                    },
                    modifier = Modifier.weight(1f),
                    maxLines = 4,
                    shape = RoundedCornerShape(24.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = PrimaryCyan,
                        unfocusedBorderColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                )

                Spacer(modifier = Modifier.width(8.dp))

                if (uiState.isGenerating) {
                    IconButton(
                        onClick = { viewModel.stopGeneration() },
                        modifier = Modifier
                            .size(48.dp)
                            .clip(CircleShape)
                            .background(Color(0xFFEF4444))
                    ) {
                        Icon(Icons.Default.Stop, contentDescription = "Stop", tint = Color.White)
                    }
                } else {
                    IconButton(
                        onClick = {
                            if (inputText.isNotBlank()) {
                                val text = inputText
                                inputText = ""
                                viewModel.sendMessage(text)
                            }
                        },
                        enabled = inputText.isNotBlank(),
                        modifier = Modifier
                            .size(48.dp)
                            .clip(CircleShape)
                            .background(if (inputText.isNotBlank()) PrimaryCyan else SurfaceLightDark)
                    ) {
                        Icon(
                            Icons.Default.Send,
                            contentDescription = "Send",
                            tint = if (inputText.isNotBlank()) Color.Black else Color.Gray
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun MessageBubble(message: ChatMessageEntity, onCopy: () -> Unit) {
    val isUser = message.role == "user"

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        if (!isUser) {
            Box(
                modifier = Modifier
                    .size(32.dp)
                    .clip(CircleShape)
                    .background(PrimaryCyan.copy(alpha = 0.2f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Default.SmartToy,
                    contentDescription = null,
                    tint = PrimaryCyan,
                    modifier = Modifier.size(18.dp)
                )
            }
            Spacer(modifier = Modifier.width(8.dp))
        }

        Surface(
            color = if (isUser) AccentIndigo else SurfaceDark,
            shape = RoundedCornerShape(
                topStart = 16.dp,
                topEnd = 16.dp,
                bottomStart = if (isUser) 16.dp else 4.dp,
                bottomEnd = if (isUser) 4.dp else 16.dp
            ),
            modifier = Modifier.widthIn(max = 300.dp)
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                if (!isUser && message.agentId.isNotBlank()) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = message.agentId.replace("_", " ").uppercase(),
                            style = MaterialTheme.typography.labelSmall,
                            color = PrimaryCyan,
                            fontWeight = FontWeight.Bold
                        )
                        if (message.tokensPerSec > 0) {
                            Text(
                                text = "%.1f tok/s".format(message.tokensPerSec),
                                style = MaterialTheme.typography.labelSmall,
                                color = TextSecondaryDark
                            )
                        }
                    }
                    Spacer(modifier = Modifier.height(4.dp))
                }

                SelectionContainer {
                    Text(
                        text = message.content,
                        style = MaterialTheme.typography.bodyMedium,
                        color = if (isUser) Color.White else TextPrimaryDark
                    )
                }

                if (!isUser) {
                    Spacer(modifier = Modifier.height(4.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.End
                    ) {
                        Icon(
                            Icons.Default.ContentCopy,
                            contentDescription = "Copy",
                            tint = TextSecondaryDark,
                            modifier = Modifier
                                .size(16.dp)
                                .clickable { onCopy() }
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun StreamingBubble(text: String, agent: AgentType) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Start
    ) {
        Box(
            modifier = Modifier
                .size(32.dp)
                .clip(CircleShape)
                .background(PrimaryCyan.copy(alpha = 0.2f)),
            contentAlignment = Alignment.Center
        ) {
            Icon(Icons.Default.Bolt, contentDescription = null, tint = PrimaryCyan, modifier = Modifier.size(18.dp))
        }
        Spacer(modifier = Modifier.width(8.dp))

        Surface(
            color = SurfaceDark,
            shape = RoundedCornerShape(16.dp, 16.dp, 16.dp, 4.dp),
            modifier = Modifier.widthIn(max = 300.dp)
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(
                    text = agent.displayName.uppercase(),
                    style = MaterialTheme.typography.labelSmall,
                    color = AccentEmerald,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = text,
                    style = MaterialTheme.typography.bodyMedium,
                    color = TextPrimaryDark
                )
            }
        }
    }
}

@Composable
fun EmptyChatSuggestions(activeAgent: AgentType, onSelectPrompt: (String) -> Unit) {
    val prompts = when (activeAgent) {
        AgentType.ORCHESTRATOR -> listOf(
            "Summarize the key architectural patterns of SLMs",
            "Plan a 3-phase rollout for our offline mobile application",
            "Calculate 1450 * 0.18 for offline tax computation"
        )
        AgentType.SUMMARIZER -> listOf(
            "Summarize this project status: We completed Phase 1 backend, tested NDK bindings, and will launch next week.",
            "Give me a 3-bullet TL;DR of modern edge AI architectures."
        )
        AgentType.TASK_PLANNER -> listOf(
            "Create a step-by-step plan to integrate offline SQLite vector search into an Android app.",
            "Break down the migration of an on-device database schema."
        )
        AgentType.RAG -> listOf(
            "What are the main requirements stated in my uploaded document?",
            "Search my notes for the project milestone deadlines."
        )
        AgentType.LOCAL_TOOLS -> listOf(
            "Calculate 2048 * 4 / 1024",
            "Evaluate 3.14159 * 25"
        )
        AgentType.GENERAL -> listOf(
            "How does on-device quantization (Q4_K_M) reduce RAM usage?",
            "Explain the difference between CPU NEON and GPU Vulkan inference."
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            Icons.Default.SmartToy,
            contentDescription = null,
            tint = PrimaryCyan,
            modifier = Modifier.size(56.dp)
        )
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = "Offline SLM Agent",
            style = MaterialTheme.typography.titleLarge,
            color = MaterialTheme.colorScheme.onBackground
        )
        Text(
            text = "Select a prompt to test offline generation:",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(modifier = Modifier.height(20.dp))

        prompts.forEach { prompt ->
            Surface(
                color = SurfaceDark,
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp)
                    .clickable { onSelectPrompt(prompt) }
            ) {
                Text(
                    text = "“$prompt”",
                    style = MaterialTheme.typography.bodySmall,
                    color = PrimaryCyan,
                    modifier = Modifier.padding(12.dp)
                )
            }
        }
    }
}
