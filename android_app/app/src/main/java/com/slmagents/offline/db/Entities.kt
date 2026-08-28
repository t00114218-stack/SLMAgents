package com.slmagents.offline.db

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(tableName = "conversations")
data class ConversationEntity(
    @PrimaryKey val id: String,
    val title: String,
    val createdAt: Long = System.currentTimeMillis(),
    val activeAgentId: String = "auto_orchestrator"
)

@Entity(
    tableName = "chat_messages",
    foreignKeys = [
        ForeignKey(
            entity = ConversationEntity::class,
            parentColumns = ["id"],
            childColumns = ["conversationId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index("conversationId")]
)
data class ChatMessageEntity(
    @PrimaryKey val id: String,
    val conversationId: String,
    val role: String, // "user", "assistant", "system", "tool"
    val content: String,
    val agentId: String = "general",
    val tokensPerSec: Float = 0f,
    val timestamp: Long = System.currentTimeMillis()
)

@Entity(tableName = "documents")
data class DocumentEntity(
    @PrimaryKey val id: String,
    val title: String,
    val filename: String,
    val charCount: Int,
    val chunkCount: Int,
    val uploadedAt: Long = System.currentTimeMillis()
)

@Entity(
    tableName = "document_chunks",
    foreignKeys = [
        ForeignKey(
            entity = DocumentEntity::class,
            parentColumns = ["id"],
            childColumns = ["docId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index("docId")]
)
data class DocumentChunkEntity(
    @PrimaryKey val id: String,
    val docId: String,
    val chunkIndex: Int,
    val content: String,
    val embeddingVector: String = "" // JSON array of float or empty if keyword RAG
)
