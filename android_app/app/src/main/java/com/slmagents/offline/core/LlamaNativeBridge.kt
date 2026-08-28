package com.slmagents.offline.core

import android.util.Log

interface TokenStreamCallback {
    fun onToken(token: String)
    fun onComplete(tokensPerSec: Float, totalTokens: Int)
}

object LlamaNativeBridge {
    private const val TAG = "LlamaNativeBridge"
    private var isLibraryLoaded = false

    init {
        try {
            System.loadLibrary("slm_llama_bridge")
            isLibraryLoaded = true
            Log.i(TAG, "Native library 'slm_llama_bridge' loaded successfully.")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Native library could not be loaded, using Kotlin runtime fallback: ${e.message}")
            isLibraryLoaded = false
        }
    }

    fun isNativeSupported(): Boolean = isLibraryLoaded

    external fun nativeInitModel(
        modelPath: String,
        nThreads: Int,
        nCtx: Int,
        useGpu: Boolean
    ): Boolean

    external fun nativeGenerate(
        prompt: String,
        temperature: Float,
        topP: Float,
        maxTokens: Int,
        callback: TokenStreamCallback
    )

    external fun nativeStop()

    external fun nativeFreeModel()

    external fun nativeIsLoaded(): Boolean
}
