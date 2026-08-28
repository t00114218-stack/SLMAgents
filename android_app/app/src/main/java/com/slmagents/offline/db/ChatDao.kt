package com.slmagents.offline.db

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface ChatDao {
    // Conversations
    @Query("SELECT * FROM conversations ORDER BY createdAt DESC")
    fun getAllConversations(): Flow<List<ConversationEntity>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertConversation(conversation: ConversationEntity)

    @Query("DELETE FROM conversations WHERE id = :conversationId")
    suspend fun deleteConversation(conversationId: String)

    // Messages
    @Query("SELECT * FROM chat_messages WHERE conversationId = :conversationId ORDER BY timestamp ASC")
    fun getMessagesForConversation(conversationId: String): Flow<List<ChatMessageEntity>>

    @Query("SELECT * FROM chat_messages WHERE conversationId = :conversationId ORDER BY timestamp ASC")
    suspend fun getMessagesList(conversationId: String): List<ChatMessageEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertMessage(message: ChatMessageEntity)

    @Query("DELETE FROM chat_messages WHERE conversationId = :conversationId")
    suspend fun deleteMessages(conversationId: String)

    // Documents
    @Query("SELECT * FROM documents ORDER BY uploadedAt DESC")
    fun getAllDocuments(): Flow<List<DocumentEntity>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertDocument(doc: DocumentEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertChunks(chunks: List<DocumentChunkEntity>)

    @Query("SELECT * FROM document_chunks")
    suspend fun getAllChunks(): List<DocumentChunkEntity>

    @Query("SELECT * FROM document_chunks WHERE docId = :docId")
    suspend fun getChunksForDocument(docId: String): List<DocumentChunkEntity>

    @Query("DELETE FROM documents WHERE id = :docId")
    suspend fun deleteDocument(docId: String)
}
