package com.slmagents.offline.ui.screens

import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.slmagents.offline.core.ModelInfo
import com.slmagents.offline.ui.MainViewModel
import com.slmagents.offline.ui.theme.*

@Composable
fun ModelScreen(viewModel: MainViewModel) {
    val uiState by viewModel.uiState.collectAsState()
    val localModels by viewModel.localModels.collectAsState()
    val context = LocalContext.current

    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            viewModel.importModelUri(it)
        }
    }

    LaunchedEffect(Unit) {
        viewModel.refreshModels()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(16.dp)
    ) {
        // Header
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column {
                Text(
                    text = "On-Device SLM Models",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Text(
                    text = "100% Offline inference via llama.cpp NDK",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            IconButton(onClick = { viewModel.refreshModels() }) {
                Icon(Icons.Default.Refresh, contentDescription = "Refresh", tint = PrimaryCyan)
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Status Banner
        uiState.statusMessage?.let { status ->
            Surface(
                color = SurfaceDark,
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 12.dp)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = status,
                        style = MaterialTheme.typography.bodySmall,
                        color = PrimaryCyan,
                        modifier = Modifier.weight(1f)
                    )
                    IconButton(
                        onClick = { viewModel.clearStatus() },
                        modifier = Modifier.size(20.dp)
                    ) {
                        Icon(Icons.Default.Close, contentDescription = "Close", tint = TextSecondaryDark, modifier = Modifier.size(14.dp))
                    }
                }
            }
        }

        // Active Model Card
        Card(
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "ACTIVE INFERENCE ENGINE",
                    style = MaterialTheme.typography.labelSmall,
                    color = PrimaryCyan,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(6.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = if (uiState.isModelLoaded) uiState.activeModelName else "No Model Active",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimaryDark,
                        modifier = Modifier.weight(1f)
                    )
                    Surface(
                        color = if (uiState.isModelLoaded) AccentEmerald.copy(alpha = 0.2f) else Color(0xFFEF4444).copy(alpha = 0.2f),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text(
                            text = if (uiState.isModelLoaded) "READY" else "UNLOADED",
                            color = if (uiState.isModelLoaded) AccentEmerald else Color(0xFFEF4444),
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                        )
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "RAM Usage: ${uiState.memoryUsageMb} MB • NDK Hardware Accelerated",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondaryDark
                    )
                    if (uiState.isModelLoaded) {
                        TextButton(
                            onClick = { viewModel.unloadModel() },
                            contentPadding = PaddingValues(0.dp)
                        ) {
                            Text("Unload", color = Color(0xFFEF4444), fontSize = 12.sp)
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Import GGUF Button
        Button(
            onClick = { filePickerLauncher.launch("*/*") },
            colors = ButtonDefaults.buttonColors(containerColor = PrimaryCyan, contentColor = Color.Black),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Icon(Icons.Default.UploadFile, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text("Import .gguf Model from Storage", fontWeight = FontWeight.Bold)
        }

        Spacer(modifier = Modifier.height(16.dp))

        LazyColumn(
            verticalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier.fillMaxSize()
        ) {
            // Local Installed Models Section
            item {
                Text(
                    text = "Downloaded / Local GGUF Models (${localModels.size})",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onBackground
                )
            }

            if (localModels.isEmpty()) {
                item {
                    Card(
                        shape = RoundedCornerShape(12.dp),
                        colors = CardDefaults.cardColors(containerColor = SurfaceDark),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(
                            modifier = Modifier.padding(16.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Icon(
                                Icons.Default.FolderOpen,
                                contentDescription = null,
                                tint = TextSecondaryDark,
                                modifier = Modifier.size(40.dp)
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "No Local .gguf Models Found",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold,
                                color = TextPrimaryDark
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = "Tap 'Import .gguf Model' above to select your downloaded .gguf file, or place .gguf files in your phone's Downloads folder and press Refresh.",
                                style = MaterialTheme.typography.bodySmall,
                                color = TextSecondaryDark
                            )
                        }
                    }
                }
            } else {
                items(localModels) { model ->
                    val isActive = uiState.isModelLoaded &&
                        (uiState.activeModelName.contains(model.name, ignoreCase = true) ||
                         model.filePath.contains(uiState.activeModelName, ignoreCase = true))

                    LocalModelCard(
                        model = model,
                        isActive = isActive,
                        onActivate = { viewModel.loadModel(model) },
                        onDelete = { viewModel.deleteLocalModel(model) }
                    )
                }
            }

            // Catalog Section
            item {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Recommended GGUF Model Catalog",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onBackground
                )
            }

            items(viewModel.modelManager.recommendedCatalog) { model ->
                RecommendedModelCard(
                    model = model,
                    onDownloadClick = {
                        model.downloadUrl?.let { url ->
                            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
                            context.startActivity(intent)
                        }
                    }
                )
            }
        }
    }
}

@Composable
fun LocalModelCard(
    model: ModelInfo,
    isActive: Boolean,
    onActivate: () -> Unit,
    onDelete: () -> Unit
) {
    Card(
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = SurfaceDark),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = model.name,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimaryDark
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "${model.formattedSize} • ${model.parameterSize} • ${model.quantization}",
                        style = MaterialTheme.typography.labelSmall,
                        color = PrimaryCyan
                    )
                }

                if (isActive) {
                    Surface(
                        color = AccentEmerald.copy(alpha = 0.2f),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text(
                            text = "ACTIVE",
                            color = AccentEmerald,
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                        )
                    }
                } else {
                    Button(
                        onClick = onActivate,
                        colors = ButtonDefaults.buttonColors(containerColor = PrimaryCyan, contentColor = Color.Black),
                        shape = RoundedCornerShape(16.dp),
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp)
                    ) {
                        Text("Activate", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    }
                }

                IconButton(onClick = onDelete) {
                    Icon(
                        Icons.Default.DeleteOutline,
                        contentDescription = "Delete",
                        tint = Color(0xFFEF4444)
                    )
                }
            }
        }
    }
}

@Composable
fun RecommendedModelCard(
    model: ModelInfo,
    onDownloadClick: () -> Unit
) {
    Card(
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = SurfaceDark),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = model.name,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimaryDark
                    )
                    Text(
                        text = "${model.parameterSize} • ${model.quantization} • ~${model.ramRequiredMb}MB RAM",
                        style = MaterialTheme.typography.labelSmall,
                        color = PrimaryCyan
                    )
                }

                if (model.downloadUrl != null) {
                    OutlinedButton(
                        onClick = onDownloadClick,
                        shape = RoundedCornerShape(16.dp),
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp)
                    ) {
                        Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(14.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Get GGUF", fontSize = 11.sp)
                    }
                }
            }

            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = model.description,
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondaryDark
            )
        }
    }
}
