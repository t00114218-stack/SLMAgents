package com.slmagents.offline.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.slmagents.offline.agents.AgentType
import com.slmagents.offline.ui.MainViewModel
import com.slmagents.offline.ui.theme.*

@Composable
fun AgentsScreen(viewModel: MainViewModel) {
    val uiState by viewModel.uiState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(16.dp)
    ) {
        Text(
            text = "Offline SLM Agent Suite",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onBackground
        )
        Text(
            text = "Specialized on-device personas that run completely isolated without cloud APIs.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(16.dp))

        LazyColumn(
            verticalArrangement = Arrangement.spacedBy(12.dp),
            modifier = Modifier.fillMaxSize()
        ) {
            items(AgentType.values()) { agent ->
                val isSelected = uiState.selectedAgent == agent
                val config = viewModel.orchestrator.getAgentConfig(agent)

                Card(
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = if (isSelected) SurfaceLightDark else SurfaceDark
                    ),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Box(
                                    modifier = Modifier
                                        .size(40.dp)
                                        .clip(RoundedCornerShape(10.dp))
                                        .background(PrimaryCyan.copy(alpha = 0.15f)),
                                    contentAlignment = Alignment.Center
                                ) {
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
                                        tint = PrimaryCyan,
                                        modifier = Modifier.size(24.dp)
                                    )
                                }
                                Spacer(modifier = Modifier.width(12.dp))
                                Column {
                                    Text(
                                        text = agent.displayName,
                                        style = MaterialTheme.typography.titleMedium,
                                        fontWeight = FontWeight.Bold,
                                        color = TextPrimaryDark
                                    )
                                    Text(
                                        text = "Temp: ${config.temperature} • Top-P: ${config.topP}",
                                        style = MaterialTheme.typography.labelSmall,
                                        color = TextSecondaryDark
                                    )
                                }
                            }

                            Button(
                                onClick = { viewModel.selectAgent(agent) },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = if (isSelected) PrimaryCyan else SurfaceDark,
                                    contentColor = if (isSelected) Color.Black else PrimaryCyan
                                ),
                                shape = RoundedCornerShape(20.dp),
                                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp)
                            ) {
                                Text(if (isSelected) "Active" else "Select", fontSize = 12.sp)
                            }
                        }

                        Spacer(modifier = Modifier.height(10.dp))
                        Text(
                            text = agent.shortDescription,
                            style = MaterialTheme.typography.bodyMedium,
                            color = TextSecondaryDark
                        )

                        Spacer(modifier = Modifier.height(8.dp))
                        Surface(
                            color = CodeBlockDark,
                            shape = RoundedCornerShape(8.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = config.systemPrompt.trim(),
                                style = MaterialTheme.typography.bodySmall,
                                color = TextSecondaryDark,
                                maxLines = 3,
                                modifier = Modifier.padding(8.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}
