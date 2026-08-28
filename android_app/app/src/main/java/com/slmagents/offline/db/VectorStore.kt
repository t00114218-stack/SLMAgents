package com.slmagents.offline.db

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.UUID
import kotlin.math.ln
import kotlin.math.sqrt

data class SearchResult(
    val chunk: DocumentChunkEntity,
    val score: Float
)

class VectorStore(private val chatDao: ChatDao) {

    suspend fun ingestDocument(
        title: String,
        filename: String,
        fullText: String,
        chunkSize: Int = 400,
        chunkOverlap: Int = 80
    ): Result<DocumentEntity> = withContext(Dispatchers.Default) {
        try {
            val docId = UUID.randomUUID().toString()
            val textChunks = splitIntoChunks(fullText, chunkSize, chunkOverlap)
            
            val chunkEntities = textChunks.mapIndexed { index, chunkText ->
                DocumentChunkEntity(
                    id = UUID.randomUUID().toString(),
                    docId = docId,
                    chunkIndex = index,
                    content = chunkText,
                    embeddingVector = "" // hybrid vector / inverted index
                )
            }

            val docEntity = DocumentEntity(
                id = docId,
                title = title,
                filename = filename,
                charCount = fullText.length,
                chunkCount = chunkEntities.size,
                uploadedAt = System.currentTimeMillis()
            )

            chatDao.insertDocument(docEntity)
            chatDao.insertChunks(chunkEntities)

            Result.success(docEntity)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun searchRelevantChunks(query: String, topK: Int = 3): List<SearchResult> = withContext(Dispatchers.Default) {
        val allChunks = chatDao.getAllChunks()
        if (allChunks.isEmpty()) return@withContext emptyList()

        val queryTerms = tokenize(query)
        if (queryTerms.isEmpty()) return@withContext emptyList()

        // BM25-like scoring across local chunks
        val scoredList = allChunks.map { chunk ->
            val chunkTerms = tokenize(chunk.content)
            val score = calculateScore(queryTerms, chunkTerms)
            SearchResult(chunk, score)
        }

        scoredList.filter { it.score > 0.05f }
            .sortedByDescending { it.score }
            .take(topK)
    }

    private fun splitIntoChunks(text: String, chunkSize: Int, overlap: Int): List<String> {
        val chunks = mutableListOf<String>()
        var start = 0
        while (start < text.length) {
            val end = (start + chunkSize).coerceAtMost(text.length)
            val chunk = text.substring(start, end).trim()
            if (chunk.isNotEmpty()) {
                chunks.add(chunk)
            }
            start += (chunkSize - overlap).coerceAtLeast(1)
        }
        return chunks
    }

    private fun tokenize(text: String): List<String> {
        return text.lowercase()
            .replace(Regex("[^a-z0-9\\s]"), " ")
            .split(Regex("\\s+"))
            .filter { it.length > 2 && it !in stopWords }
    }

    private fun calculateScore(queryTerms: List<String>, docTerms: List<String>): Float {
        if (docTerms.isEmpty()) return 0f
        var matchCount = 0f
        val docTermCounts = docTerms.groupingBy { it }.eachCount()

        for (q in queryTerms) {
            val count = docTermCounts[q] ?: 0
            if (count > 0) {
                matchCount += 1.0f + ln(count.toFloat())
            }
        }
        val docLengthNorm = sqrt(docTerms.size.toFloat())
        return if (docLengthNorm > 0) (matchCount / docLengthNorm) else 0f
    }

    companion object {
        private val stopWords = setOf(
            "the", "and", "is", "in", "to", "of", "it", "with", "as", "for", "on", "was",
            "at", "by", "an", "be", "this", "that", "from", "or", "are", "which", "will"
        )
    }
}
