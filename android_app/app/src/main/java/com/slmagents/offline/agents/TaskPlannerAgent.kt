package com.slmagents.offline.agents

class TaskPlannerAgent {

    val config = AgentConfig(
        type = AgentType.TASK_PLANNER,
        systemPrompt = """
            You are a rigorous Task Planner Agent.
            When presented with a project, technical objective, or workflow:
            1. Clearly define the end goal and success criteria.
            2. Break the work down into sequential phases (Phase 1, Phase 2, etc.).
            3. For each phase, provide specific actionable bullet points with dependencies and deliverables.
            4. Highlight potential edge cases or risks.
        """.trimIndent(),
        temperature = 0.5f,
        topP = 0.9f,
        maxTokens = 1500
    )
}
