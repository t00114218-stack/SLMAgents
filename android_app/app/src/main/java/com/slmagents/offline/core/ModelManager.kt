package com.slmagents.offline.core

import android.content.Context
import android.net.Uri
import android.os.Environment
import android.provider.OpenableColumns
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.Locale

data class ModelInfo(
    val id: String,
    val name: String,
    val sizeBytes: Long,
    val filePath: String,
    val isLocal: Boolean,
    val parameterSize: String,
    val quantization: String,
    val ramRequiredMb: Int,
    val downloadUrl: String? = null,
    val description: String = ""
) {
    val formattedSize: String
        get() = when {
            sizeBytes >= 1_073_741_824L -> String.format(Locale.US, "%.2f GB", sizeBytes / 1_073_741_824.0)
            sizeBytes >= 1_048_576L -> String.format(Locale.US, "%.1f MB", sizeBytes / 1_048_576.0)
            else -> "$sizeBytes bytes"
        }
}

class ModelManager(private val context: Context) {
    private val tag = "ModelManager"

    // App internal models directory
    val modelsDir: File
        get() = File(context.filesDir, "models").apply { if (!exists()) mkdirs() }

    val recommendedCatalog = listOf(
        ModelInfo(
            id = "qwen-2.5-0.5b-q4",
            name = "Qwen2.5 0.5B Instruct",
            sizeBytes = 398_000_000L,
            filePath = "",
            isLocal = false,
            parameterSize = "0.5 Billion",
            quantization = "Q4_K_M",
            ramRequiredMb = 450,
            downloadUrl = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
            description = "Ultra-lightweight, extremely fast on all Android phones (~40 tps). Great for agent routing & quick queries."
        ),
        ModelInfo(
            id = "smollm2-1.7b-q4",
            name = "SmolLM2 1.7B Instruct",
            sizeBytes = 1_050_000_000L,
            filePath = "",
            isLocal = false,
            parameterSize = "1.7 Billion",
            quantization = "Q4_K_M",
            ramRequiredMb = 1100,
            downloadUrl = "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct-q4_k_m.gguf",
            description = "Top tier for local reasoning, summarization, tool calling, and document RAG with minimal battery drain."
        ),
        ModelInfo(
            id = "qwen-2.5-1.5b-q4",
            name = "Qwen2.5 1.5B Instruct",
            sizeBytes = 986_000_000L,
            filePath = "",
            isLocal = false,
            parameterSize = "1.5 Billion",
            quantization = "Q4_K_M",
            ramRequiredMb = 1050,
            downloadUrl = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            description = "High accuracy multilingual and structured JSON responses for local agents."
        ),
        ModelInfo(
            id = "llama-3.2-1b-q4",
            name = "Llama 3.2 1B Instruct",
            sizeBytes = 850_000_000L,
            filePath = "",
            isLocal = false,
            parameterSize = "1.2 Billion",
            quantization = "Q4_K_M",
            ramRequiredMb = 900,
            downloadUrl = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            description = "Compact Meta Llama 3.2 model for general offline chat and task breakdown."
        )
    )

    fun listLocalModels(): List<ModelInfo> {
        val list = mutableListOf<ModelInfo>()

        // Search internal app models dir
        modelsDir.listFiles()?.filter { it.extension.lowercase() == "gguf" }?.forEach { file ->
            list.add(
                ModelInfo(
                    id = file.absolutePath,
                    name = file.nameWithoutExtension.replace("-", " ").replace("_", " ").capitalizeWords(),
                    sizeBytes = file.length(),
                    filePath = file.absolutePath,
                    isLocal = true,
                    parameterSize = parseParameterSize(file.name),
                    quantization = parseQuantization(file.name),
                    ramRequiredMb = (file.length() / (1024 * 1024) * 1.25).toInt()
                )
            )
        }

        // Search standard Downloads folder if readable
        try {
            val downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            if (downloads != null && downloads.exists() && downloads.canRead()) {
                downloads.listFiles()?.filter { it.extension.lowercase() == "gguf" }?.forEach { file ->
                    if (list.none { it.filePath == file.absolutePath }) {
                        list.add(
                            ModelInfo(
                                id = file.absolutePath,
                                name = "[Downloads] " + file.nameWithoutExtension.replace("-", " ").replace("_", " ").capitalizeWords(),
                                sizeBytes = file.length(),
                                filePath = file.absolutePath,
                                isLocal = true,
                                parameterSize = parseParameterSize(file.name),
                                quantization = parseQuantization(file.name),
                                ramRequiredMb = (file.length() / (1024 * 1024) * 1.25).toInt()
                            )
                        )
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(tag, "Could not access Downloads folder: ${e.message}")
        }

        return list
    }

    suspend fun importModelFromUri(uri: Uri, overrideName: String? = null): Result<File> = withContext(Dispatchers.IO) {
        try {
            val fileName = overrideName ?: resolveFileName(uri) ?: "imported_model_${System.currentTimeMillis()}.gguf"
            val sanitizedFileName = if (fileName.lowercase().endsWith(".gguf")) fileName else "$fileName.gguf"
            val targetFile = File(modelsDir, sanitizedFileName)

            context.contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(targetFile).use { output ->
                    input.copyTo(output)
                }
            } ?: return@withContext Result.failure(Exception("Cannot open stream for URI: $uri"))

            Log.i(tag, "Imported GGUF model successfully to: ${targetFile.absolutePath}")
            Result.success(targetFile)
        } catch (e: Exception) {
            Log.e(tag, "Failed to import model: ${e.message}", e)
            Result.failure(e)
        }
    }

    fun deleteModelFile(filePath: String): Boolean {
        return try {
            val file = File(filePath)
            if (file.exists() && file.canWrite()) {
                file.delete()
            } else false
        } catch (e: Exception) {
            Log.e(tag, "Failed to delete file $filePath: ${e.message}")
            false
        }
    }

    private fun resolveFileName(uri: Uri): String? {
        var result: String? = null
        if (uri.scheme == "content") {
            try {
                context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                    if (cursor.moveToFirst()) {
                        val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                        if (nameIndex != -1) {
                            result = cursor.getString(nameIndex)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.w(tag, "Error querying content resolver: ${e.message}")
            }
        }
        if (result == null) {
            result = uri.path?.let { path ->
                val cut = path.lastIndexOf('/')
                if (cut != -1) path.substring(cut + 1) else path
            }
        }
        return result
    }

    private fun parseParameterSize(filename: String): String {
        val lower = filename.lowercase()
        return when {
            lower.contains("0.5b") -> "0.5B"
            lower.contains("360m") -> "360M"
            lower.contains("1.5b") -> "1.5B"
            lower.contains("1.7b") -> "1.7B"
            lower.contains("1b") -> "1B"
            lower.contains("3b") -> "3B"
            lower.contains("7b") -> "7B"
            else -> "SLM"
        }
    }

    private fun parseQuantization(filename: String): String {
        val lower = filename.lowercase()
        return when {
            lower.contains("q4_k_m") -> "Q4_K_M"
            lower.contains("q4_0") -> "Q4_0"
            lower.contains("q5_k_m") -> "Q5_K_M"
            lower.contains("q8_0") -> "Q8_0"
            lower.contains("f16") -> "F16"
            else -> "Quantized"
        }
    }

    private fun String.capitalizeWords(): String =
        split(" ").joinToString(" ") { it.replaceFirstChar { char -> char.uppercase() } }
}
