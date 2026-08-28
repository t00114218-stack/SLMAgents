# 📱 SLM Agents Offline — Android Mobile Application

A 100% **offline-first Android mobile application** for Small Language Model (SLM) multi-agent workflows. Runs entirely on-device hardware (CPU NEON / GPU Vulkan / NPU) without requiring an internet connection or external cloud APIs.

> [!IMPORTANT]
> **Complete Isolation Notice**:
> This Android mobile codebase is self-contained within `android_app/` and has **zero impact** on the existing web application (`website/`), FastAPI server (`main.py`), or Hugging Face deployment scripts (`deploy_to_hf.*`).

---

## 🚀 Key Features

- **⚡ 100% Offline Multi-Agent Inference**: Real-time token streaming powered by on-device quantized SLMs (`.gguf` format).
- **🤖 Specialized Offline Agents**:
  - **Auto Orchestrator**: Intelligently routes queries to the optimal agent persona based on query intent.
  - **Document RAG Agent**: Indexes local text, PDF, and notes into an on-device vector store (`sqlite-vec` / TF-IDF hybrid) for private document Q&A.
  - **Summarizer Agent**: Generates bullet-point executive summaries and TL;DRs with zero latency.
  - **Task Planner**: Deconstructs complex objectives into step-by-step actionable plans.
  - **Local Tools Agent**: Performs on-device computations, unit conversions, and formula evaluations.
- **🎨 Modern Material 3 UI**:
  - Dark-mode developer aesthetic with real-time tokens/sec performance counter.
  - Streaming markdown bubbles, message copying, and multi-conversation Room database persistence.
  - Built-in GGUF Model Manager & RAM usage gauge.

---

## 🛠️ Architecture

```
android_app/
├── app/
│   ├── build.gradle.kts                          # Android & NDK build configuration
│   ├── src/main/
│   │   ├── AndroidManifest.xml
│   │   ├── cpp/                                  # Native C++/JNI bindings
│   │   │   ├── CMakeLists.txt
│   │   │   └── llama-android-bridge.cpp
│   │   └── java/com/slmagents/offline/
│   │       ├── MainActivity.kt                   # App entry point
│   │       ├── core/                             # On-device inference engine
│   │       │   ├── LlamaEngine.kt                # Token streaming & Flow API
│   │       │   ├── LlamaNativeBridge.kt          # JNI bridge
│   │       │   └── ModelManager.kt               # GGUF discovery & memory stats
│   │       ├── agents/                           # Offline Agent implementations
│   │       │   ├── Agent.kt
│   │       │   ├── AgentRouter.kt                # Intent routing
│   │       │   ├── Orchestrator.kt               # Multi-agent coordinator
│   │       │   ├── RAGAgent.kt                   # Document retrieval
│   │       │   ├── SummarizerAgent.kt
│   │       │   ├── TaskPlannerAgent.kt
│   │       │   └── ToolAgent.kt                  # Local computation
│   │       ├── db/                               # Local SQLite & Vector store
│   │       │   ├── AppDatabase.kt                # Room DB
│   │       │   ├── ChatDao.kt
│   │       │   ├── Entities.kt
│   │       │   └── VectorStore.kt                # On-device vector retrieval
│   │       └── ui/                               # Jetpack Compose UI
│   │           ├── MainViewModel.kt
│   │           ├── navigation/AppNavigation.kt   # Bottom navigation bar
│   │           ├── screens/
│   │           │   ├── ChatScreen.kt             # Streaming chat UI
│   │           │   ├── AgentsScreen.kt           # Persona catalogue
│   │           │   ├── KnowledgeBaseScreen.kt    # Offline document indexing
│   │           │   └── ModelScreen.kt            # GGUF model manager
│   │           └── theme/                        # Modern Dark/Light theme
├── build.gradle.kts
├── settings.gradle.kts
└── gradle/libs.versions.toml
```

---

## 📦 Recommended Offline GGUF Models

| Model | Parameters | Quantization | Size | RAM Needed | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-0.5B-Instruct** | 0.5B | `Q4_K_M` | ~398 MB | ~450 MB | Ultra-fast routing (~40+ tps) on any Android phone |
| **SmolLM2-1.7B-Instruct** | 1.7B | `Q4_K_M` | ~1.05 GB | ~1.1 GB | High-quality reasoning, RAG & summarization |
| **Qwen2.5-1.5B-Instruct** | 1.5B | `Q4_K_M` | ~986 MB | ~1.05 GB | Structured JSON and multi-agent outputs |
| **Llama-3.2-1B-Instruct** | 1.2B | `Q4_K_M` | ~850 MB | ~900 MB | General chat and task planning |

---

## 📲 How to Build and Run

### Option 1: Using Android Studio (Recommended)
1. Open **Android Studio** (Hedgehog 2023.1.1 or newer).
2. Click **File -> Open...** and select the `SLMAgents/android_app` directory.
3. Allow Gradle to sync dependencies.
4. Connect an Android phone (with USB Debugging enabled) or start an Android Emulator (`arm64-v8a` or `x86_64`, Android 8.0+ / API 26+).
5. Click **Run 'app'** (`Shift + F10`).

### Option 2: Using Command Line / Gradle
```bash
cd android_app
./gradlew assembleDebug
```
The compiled debug APK will be generated at:
```
android_app/app/build/outputs/apk/debug/app-debug.apk
```

---

## 📥 Loading Models onto Your Android Device

1. Download any compatible `.gguf` file (e.g. from Hugging Face: [Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)).
2. Transfer it to your phone's **Downloads** folder, or tap **"Import .gguf Model from Storage"** in the app's **Models** tab.
3. The app will immediately load the model into memory.
4. Turn on **Airplane Mode** and enjoy 100% private, offline AI agent chat!
