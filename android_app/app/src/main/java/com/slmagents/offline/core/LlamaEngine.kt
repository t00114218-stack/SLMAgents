package com.slmagents.offline.core

import android.content.Context
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File

data class GenerationStats(
    val tokensPerSecond: Float = 0f,
    val totalTokens: Int = 0,
    val latencyMs: Long = 0,
    val memoryUsageMb: Long = 0
)

data class InferenceConfig(
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val maxTokens: Int = 1024,
    val threads: Int = 4,
    val contextSize: Int = 2048,
    val useGpu: Boolean = true
)

class LlamaEngine(private val context: Context) {
    private val tag = "LlamaEngine"
    private var loadedModelPath: String? = null
    var currentStats: GenerationStats = GenerationStats()
        private set

    val isModelLoaded: Boolean
        get() = loadedModelPath != null && (
            if (LlamaNativeBridge.isNativeSupported()) LlamaNativeBridge.nativeIsLoaded() else true
        )

    suspend fun loadModel(
        modelFile: File,
        config: InferenceConfig = InferenceConfig()
    ): Result<Boolean> = withContext(Dispatchers.IO) {
        if (!modelFile.exists()) {
            return@withContext Result.failure(IllegalArgumentException("Model file does not exist at: ${modelFile.absolutePath}"))
        }

        try {
            if (LlamaNativeBridge.isNativeSupported()) {
                val success = LlamaNativeBridge.nativeInitModel(
                    modelPath = modelFile.absolutePath,
                    nThreads = config.threads,
                    nCtx = config.contextSize,
                    useGpu = config.useGpu
                )
                if (success) {
                    loadedModelPath = modelFile.absolutePath
                    Log.i(tag, "Loaded model successfully via native runtime: ${modelFile.name}")
                    return@withContext Result.success(true)
                } else {
                    return@withContext Result.failure(RuntimeException("Native model initialization failed"))
                }
            } else {
                loadedModelPath = modelFile.absolutePath
                Log.i(tag, "Loaded model path in standalone fallback mode: ${modelFile.name}")
                return@withContext Result.success(true)
            }
        } catch (e: Exception) {
            Log.e(tag, "Error loading model: ${e.message}", e)
            return@withContext Result.failure(e)
        }
    }

    fun generateStream(
        prompt: String,
        config: InferenceConfig = InferenceConfig()
    ): Flow<String> = callbackFlow {
        if (!isModelLoaded) {
            trySend("[Error: No offline model loaded. Please select or import a .gguf model in Model Settings.]")
            close()
            return@callbackFlow
        }

        val startTime = SystemClock.elapsedRealtime()
        var tokenCount = 0

        val callback = object : TokenStreamCallback {
            override fun onToken(token: String) {
                if (token.isNotEmpty()) {
                    tokenCount++
                    trySend(token)
                }
            }

            override fun onComplete(tokensPerSec: Float, totalTokens: Int) {
                val elapsed = SystemClock.elapsedRealtime() - startTime
                val mem = getUsedMemoryMb()
                currentStats = GenerationStats(
                    tokensPerSecond = if (tokensPerSec > 0) tokensPerSec else if (elapsed > 0) (tokenCount * 1000f / elapsed) else 0f,
                    totalTokens = if (totalTokens > 0) totalTokens else tokenCount,
                    latencyMs = elapsed,
                    memoryUsageMb = mem
                )
                close()
            }
        }

        if (LlamaNativeBridge.isNativeSupported()) {
            LlamaNativeBridge.nativeGenerate(
                prompt = prompt,
                temperature = config.temperature,
                topP = config.topP,
                maxTokens = config.maxTokens,
                callback = callback
            )
        } else {
            // Emulated local token stream for UI development without physical GGUF
            val sampleResponse = "This is an offline response generated locally by the SLM Agent engine."
            val words = sampleResponse.split(" ")
            for (word in words) {
                tokenCount++
                trySend("$word ")
                Thread.sleep(40)
            }
            callback.onComplete(25.0f, tokenCount)
        }

        awaitClose {
            if (LlamaNativeBridge.isNativeSupported()) {
                LlamaNativeBridge.nativeStop()
            }
        }
    }.flowOn(Dispatchers.IO)

    fun stopGeneration() {
        if (LlamaNativeBridge.isNativeSupported()) {
            LlamaNativeBridge.nativeStop()
        }
    }

    fun unloadModel() {
        if (LlamaNativeBridge.isNativeSupported()) {
            LlamaNativeBridge.nativeFreeModel()
        }
        loadedModelPath = null
    }

    fun getUsedMemoryMb(): Long {
        val runtime = Runtime.getRuntime()
        val usedMemInBytes = runtime.totalMemory() - runtime.freeMemory()
        return usedMemInBytes / (1024 * 1024)
    }

    companion object {
        /**
         * Formats conversation into ChatML prompt standard for Qwen2.5 & SmolLM2
         */
        fun formatChatML(systemPrompt: String, history: List<Pair<String, String>>, userMessage: String): String {
            val sb = java.lang.StringBuilder()
            if (systemPrompt.isNotEmpty()) {
                sb.append("<|im_start|>system\n").append(systemPrompt.trim()).append("<|im_end|>\n")
            }
            for ((role, content) in history) {
                sb.append("<|im_start|>").append(role).append("\n").append(content.trim()).append("<|im_end|>\n")
            }
            sb.append("<|im_start|>user\n").append(userMessage.trim()).append("<|im_end|>\n")
            sb.append("<|im_start|>assistant\n")
            return sb.toString()
        }
    }
}
