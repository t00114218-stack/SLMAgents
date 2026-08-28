package com.slmagents.offline.agents

import android.util.Log

class ToolAgent {

    val config = AgentConfig(
        type = AgentType.LOCAL_TOOLS,
        systemPrompt = """
            You are a precise Local Tool & Computation Agent.
            You excel at calculations, unit conversions, data transformations, and structured data extraction.
            Present formulas, step-by-step logic, and final computed answers cleanly.
        """.trimIndent(),
        temperature = 0.2f,
        topP = 0.8f,
        maxTokens = 800
    )

    fun executeLocalMath(expression: String): String? {
        return try {
            val clean = expression.replace(" ", "")
            val result = evaluateSimpleExpression(clean)
            result?.toString()
        } catch (e: Exception) {
            Log.e("ToolAgent", "Error evaluating math: ${e.message}")
            null
        }
    }

    private fun evaluateSimpleExpression(expr: String): Double? {
        val regex = Regex("^([0-9.]+)([+\\-*/^])([0-9.]+)$")
        val match = regex.find(expr) ?: return null
        val (num1Str, op, num2Str) = match.destructured
        val a = num1Str.toDoubleOrNull() ?: return null
        val b = num2Str.toDoubleOrNull() ?: return null

        return when (op) {
            "+" -> a + b
            "-" -> a - b
            "*" -> a * b
            "/" -> if (b != 0.0) a / b else Double.NaN
            "^" -> Math.pow(a, b)
            else -> null
        }
    }
}
