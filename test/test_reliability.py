import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for package in (
    "slm_orchestrator",
    "slm_pdf",
    "slm_rag",
    "slm_task_planner",
    "slm_text_to_sql",
    "slm_web_agent",
    "slm_code_interpreter",
):
    sys.path.insert(0, os.path.join(ROOT, package))


class ReliabilityTests(unittest.TestCase):
    def test_execute_plan_adds_bounded_code_handoff(self):
        from slm_orchestrator import SLMOrchestrator

        orchestrator = object.__new__(SLMOrchestrator)
        pipeline = orchestrator._detect_agent_pipeline(
            "Execute this plan and build the application", "SLMTaskPlanner"
        )

        self.assertEqual(pipeline, ["SLMTaskPlanner", "SLMCodeInterpreter"])

    def test_planner_fallback_is_goal_specific_and_structured(self):
        from slm_task_planner import SLMTaskPlanner

        planner = object.__new__(SLMTaskPlanner)
        planner.model = None
        planner.tokenizer = None
        result = planner.build_plan("build a weather station")

        self.assertIn(result["status"], ["degraded", "success"])
        self.assertEqual(result["total_steps"], 3)
        self.assertIn("weather station", result["plan_markdown"])
        self.assertNotIn("App Store", result["plan_markdown"])

    def test_planner_routes_pdf_extraction_without_model(self):
        from slm_task_planner import SLMTaskPlanner

        planner = object.__new__(SLMTaskPlanner)
        planner.model = None
        planner.tokenizer = None
        result = planner.build_plan("Extract stats from PDF")

        self.assertEqual(result["total_steps"], 1)
        self.assertIn("SLMPDFChat", result["tasks"][0]["assigned_agent"])

    def test_rag_refuses_to_self_ground_on_question(self):
        from slm_rag import SLMRag

        rag = object.__new__(SLMRag)
        answer = rag.query("The moon is made of cheese", chunks=[])

        self.assertIn("no document context", answer.lower())

    def test_sql_validation_fails_closed_for_unknown_table(self):
        from slm_text_to_sql import SLMTextToSQL

        valid, error = SLMTextToSQL._validate_sql(
            "CREATE TABLE users (id INTEGER);", "SELECT * FROM invented;"
        )

        self.assertFalse(valid)
        self.assertIn("no such table", error.lower())

    def test_pdf_extraction_failure_does_not_invent_content(self):
        from slm_pdf import SLMPDFChat

        pdf = object.__new__(SLMPDFChat)
        pdf.doc_parser = None
        pdf.loaded_chunks = []
        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            result = pdf.load(handle.name)

        self.assertFalse(result["success"])
        self.assertEqual(pdf.loaded_chunks, [])

    def test_browser_dependency_failure_is_not_success(self):
        from slm_web_agent import SLMWebAgent

        browser = object.__new__(SLMWebAgent)
        browser.page = None
        browser.start_browser = lambda: False
        result = browser.browse("buy an item", "https://example.com")

        self.assertFalse(result["success"])
        self.assertEqual(result["finish_reason"], "dependency_unavailable")

    def test_code_generation_stops_at_complete_fence(self):
        from slm_code_interpreter import SLMCodeInterpreter

        self.assertFalse(SLMCodeInterpreter._generation_complete("```python\nprint(1)"))
        self.assertTrue(SLMCodeInterpreter._generation_complete("```python\nprint(1)\n```"))


if __name__ == "__main__":
    unittest.main()
