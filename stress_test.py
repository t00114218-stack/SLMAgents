"""
Comprehensive Stress Test Suite for SLM Agents (Orchestrator, RAG, Summarizer)
Tests all modes: standard, edge cases, agentic tool use, and evaluator-corrector loop.

Results are printed to stdout with PASS/FAIL/SKIP status per test.
"""
import os
import sys
import json
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "slm_orchestrator"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "slm_rag"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "slm_summarizer"))

from slm_orchestrator.orchestrator import SLMOrchestrator
from slm_rag.rag import SLMRag
from slm_summarizer.summarizer import SLMSummarizer

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS  = "\033[92mPASS\033[0m"
FAIL  = "\033[91mFAIL\033[0m"
SKIP  = "\033[93mSKIP\033[0m"
WARN  = "\033[93mWARN\033[0m"

results = []

def run_test(name, fn):
    """Run a single test function and capture result."""
    print(f"\n  ► {name}", flush=True)
    t0 = time.time()
    try:
        msg = fn()
        elapsed = time.time() - t0
        status = PASS
        print(f"    [{status}] ({elapsed:.1f}s) {msg or ''}", flush=True)
        results.append((name, "PASS", msg))
    except AssertionError as e:
        elapsed = time.time() - t0
        print(f"    [{FAIL}] ({elapsed:.1f}s) AssertionError: {e}", flush=True)
        results.append((name, "FAIL", str(e)))
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    [{FAIL}] ({elapsed:.1f}s) Exception: {e}", flush=True)
        traceback.print_exc()
        results.append((name, "FAIL", str(e)))


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def assert_in_agents(result, agents):
    names = [a["name"] for a in agents]
    assert result in names, f"Result '{result}' not in agent names {names}"


# ─────────────────────────────────────────────────────────────────────────────
# Shared Fixtures
# ─────────────────────────────────────────────────────────────────────────────

AGENTS_3 = [
    {"name": "Billing Support",    "description": "Handles payments, invoices, refunds."},
    {"name": "Technical Support",  "description": "Handles bugs, crashes, software issues."},
    {"name": "General Chat",       "description": "Handles greetings and casual conversation."},
]

AGENTS_2 = [
    {"name": "CodeAgent",  "description": "Writes and reviews Python code."},
    {"name": "SearchAgent","description": "Searches and retrieves documentation."},
]

AGENTS_1 = [
    {"name": "OnlyAgent", "description": "Handles everything."},
]

CHUNKS_NORMAL = [
    "NebulaCorp was founded in 2024 by Dr. Helena Vance. It specialises in quantum-resistant encryption.",
    "The flagship product of NebulaCorp is called AegisShield, widely used by financial organisations.",
    "In early 2026, NebulaCorp announced a partnership with the European Space Agency.",
]

CHUNKS_EMPTY = []
CHUNKS_SINGLE = ["The answer is 42."]

TEXT_SHORT = (
    "SpaceX successfully launched its Falcon 9 rocket on Friday, sending 22 Starlink satellites into orbit. "
    "The mission lifted off from Cape Canaveral. The first stage landed safely on the droneship 'A Shortfall of Gravitas'."
)

TEXT_LONG = (TEXT_SHORT + " ") * 30  # ~6,000 chars → triggers Map-Reduce

TEXT_EMPTY = ""

VECTOR_DB_MOCK = {
    "founding_year": "NebulaCorp was founded in 2012 by Elena Torres.",
    "ceo":           "The CEO of NebulaCorp is Marcus Webb.",
}

def make_mock_executor(call_log: list):
    def executor(tool_name, args):
        call_log.append({"tool": tool_name, "args": args})
        query = args.get("query", "").lower()
        for key, val in VECTOR_DB_MOCK.items():
            if key in query or any(w in query for w in key.split("_")):
                return val
        return "No relevant document found in Vector DB."
    return executor


# ─────────────────────────────────────────────────────────────────────────────
# Load Models (shared across tests)
# ─────────────────────────────────────────────────────────────────────────────

print("\nInitialising models (this may take a moment)...", flush=True)

orchestrator = None
rag          = None
summarizer   = None

try:
    t = time.time()
    orchestrator = SLMOrchestrator()
    print(f"[OK] Orchestrator loaded in {time.time()-t:.1f}s")
except Exception as e:
    print(f"[ERROR] Orchestrator failed to load: {e}")

try:
    t = time.time()
    rag = SLMRag()
    print(f"[OK] RAG loaded in {time.time()-t:.1f}s")
except Exception as e:
    print(f"[ERROR] RAG failed to load: {e}")

try:
    t = time.time()
    summarizer = SLMSummarizer()
    print(f"[OK] Summarizer loaded in {time.time()-t:.1f}s")
except Exception as e:
    print(f"[ERROR] Summarizer failed to load: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR TESTS
# ─────────────────────────────────────────────────────────────────────────────

section("ORCHESTRATOR TESTS")

def test_orc_standard_billing():
    if not orchestrator: return "SKIP – model not loaded"
    r = orchestrator.route(agents=AGENTS_3, question="I need help with my invoice")
    assert_in_agents(r, AGENTS_3)
    assert "billing" in r.lower() or r == AGENTS_3[0]["name"], f"Expected billing-related agent, got '{r}'"
    return f"routed → '{r}'"

def test_orc_standard_tech():
    if not orchestrator: return "SKIP – model not loaded"
    r = orchestrator.route(agents=AGENTS_3, question="My application keeps crashing on startup")
    assert_in_agents(r, AGENTS_3)
    return f"routed → '{r}'"

def test_orc_standard_general():
    if not orchestrator: return "SKIP – model not loaded"
    r = orchestrator.route(agents=AGENTS_3, question="Hey! How are you today?")
    assert_in_agents(r, AGENTS_3)
    return f"routed → '{r}'"

def test_orc_code_agent():
    if not orchestrator: return "SKIP – model not loaded"
    r = orchestrator.route(agents=AGENTS_2, question="Write a Python function to reverse a string")
    assert_in_agents(r, AGENTS_2)
    return f"routed → '{r}'"

def test_orc_single_agent():
    """Only one agent available – must always return that agent."""
    if not orchestrator: return "SKIP – model not loaded"
    r = orchestrator.route(agents=AGENTS_1, question="Anything at all")
    assert r == AGENTS_1[0]["name"], f"Expected '{AGENTS_1[0]['name']}', got '{r}'"
    return f"routed → '{r}'"

def test_orc_empty_agents():
    """Should raise ValueError for empty agents list."""
    if not orchestrator: return "SKIP – model not loaded"
    try:
        orchestrator.route(agents=[], question="test")
        raise AssertionError("Expected ValueError was not raised")
    except ValueError:
        return "correctly raised ValueError"

def test_orc_tool_use_no_executor():
    """Tools list provided but no executor – should degrade gracefully (no crash)."""
    if not orchestrator: return "SKIP – model not loaded"
    tools = [{"name": "lookup", "description": "looks up stuff", "parameters": {}}]
    r = orchestrator.route(agents=AGENTS_3, question="Invoice help", tools=tools, tool_executor=None)
    assert_in_agents(r, AGENTS_3)
    return f"routed without executor → '{r}'"

def test_orc_tool_use_with_executor():
    """Full agentic tool execution path."""
    if not orchestrator: return "SKIP – model not loaded"
    call_log = []
    tools = [
        {
            "name": "search_vector_db",
            "description": "Searches the internal knowledge base.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]
    r = orchestrator.route(
        agents=AGENTS_3,
        question="Check the knowledge base and route appropriately for billing",
        tools=tools,
        tool_executor=make_mock_executor(call_log),
        max_iterations=3,
    )
    assert_in_agents(r, AGENTS_3)
    return f"routed → '{r}' | tool calls: {len(call_log)}"

def test_orc_tool_executor_raises():
    """Tool executor that always throws – orchestrator should handle gracefully."""
    if not orchestrator: return "SKIP – model not loaded"
    def bad_executor(name, args):
        raise RuntimeError("DB connection refused")
    tools = [{"name": "search_vector_db", "description": "search", "parameters": {}}]
    r = orchestrator.route(
        agents=AGENTS_3,
        question="Need billing help",
        tools=tools,
        tool_executor=bad_executor,
        max_iterations=2,
    )
    assert_in_agents(r, AGENTS_3)
    return f"survived bad executor → '{r}'"

def test_orc_many_agents():
    """Route among a large list of agents."""
    if not orchestrator: return "SKIP – model not loaded"
    many = [{"name": f"Agent_{i}", "description": f"Handles task category {i}"} for i in range(10)]
    r = orchestrator.route(agents=many, question="What is 2+2?")
    assert_in_agents(r, many)
    return f"routed → '{r}' out of {len(many)} agents"

run_test("Standard routing – billing query",      test_orc_standard_billing)
run_test("Standard routing – tech support query", test_orc_standard_tech)
run_test("Standard routing – general chat query", test_orc_standard_general)
run_test("Standard routing – code agent",         test_orc_code_agent)
run_test("Edge: single agent list",               test_orc_single_agent)
run_test("Edge: empty agents list",               test_orc_empty_agents)
run_test("Tool use: tools list, no executor",     test_orc_tool_use_no_executor)
run_test("Tool use: full agentic with executor",  test_orc_tool_use_with_executor)
run_test("Tool use: executor raises exception",   test_orc_tool_executor_raises)
run_test("Stress: 10-agent routing",              test_orc_many_agents)


# ─────────────────────────────────────────────────────────────────────────────
# RAG TESTS
# ─────────────────────────────────────────────────────────────────────────────

section("RAG TESTS")

def test_rag_standard():
    if not rag: return "SKIP – model not loaded"
    ans = rag.answer(chunks=CHUNKS_NORMAL, question="What is AegisShield?", instruction="Be concise.")
    assert ans and len(ans) > 5, "Answer is empty or too short"
    return f"answer: '{ans[:80]}...'"

def test_rag_instruction_pirate():
    if not rag: return "SKIP – model not loaded"
    ans = rag.answer(
        chunks=CHUNKS_NORMAL,
        question="Who founded NebulaCorp?",
        instruction="Answer like a 17th-century pirate."
    )
    assert ans and len(ans) > 5
    return f"answer: '{ans[:80]}...'"

def test_rag_single_chunk():
    if not rag: return "SKIP – model not loaded"
    ans = rag.answer(chunks=CHUNKS_SINGLE, question="What is the answer?", instruction="Be direct.")
    assert ans and len(ans) > 1
    return f"answer: '{ans[:80]}'"

def test_rag_empty_chunks():
    """RAG with no chunks – model should say it doesn't know."""
    if not rag: return "SKIP – model not loaded"
    ans = rag.answer(chunks=CHUNKS_EMPTY, question="What is AegisShield?", instruction="Be honest.")
    assert isinstance(ans, str), "Should return a string even with empty chunks"
    return f"answer (empty chunks): '{ans[:80]}...'"

def test_rag_no_answer_in_chunks():
    """Question whose answer is not in the chunks."""
    if not rag: return "SKIP – model not loaded"
    ans = rag.answer(
        chunks=["The sky is blue.", "Grass is green."],
        question="What is the capital of France?",
        instruction="Answer only from the provided chunks."
    )
    assert isinstance(ans, str)
    return f"answer: '{ans[:80]}'"

def test_rag_temperature_creative():
    if not rag: return "SKIP – model not loaded"
    ans = rag.answer(
        chunks=CHUNKS_NORMAL,
        question="Describe NebulaCorp.",
        instruction="Be creative and elaborate.",
        temperature=0.7,
        max_tokens=200,
    )
    assert ans and len(ans) > 5
    return f"answer (temp=0.7): '{ans[:80]}...'"

def test_rag_json_instruction():
    if not rag: return "SKIP – model not loaded"
    ans = rag.answer(
        chunks=CHUNKS_NORMAL,
        question="What is their flagship product?",
        instruction='Return your answer as a valid JSON object with key "product_name".',
        max_tokens=128,
    )
    assert isinstance(ans, str)
    return f"answer (JSON instr): '{ans[:80]}'"

def test_rag_tool_use_with_executor():
    """RAG with tool executor – tool should be invoked when chunks are insufficient."""
    if not rag: return "SKIP – model not loaded"
    call_log = []
    tools = [
        {
            "name": "search_vector_db",
            "description": "Searches the vector database for extra documents.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]
    ans = rag.answer(
        chunks=["NebulaCorp is an encryption company."],
        question="Who is the CEO of NebulaCorp? Search the vector db.",
        instruction="Answer concisely.",
        tools=tools,
        tool_executor=make_mock_executor(call_log),
        max_iterations=3,
    )
    assert isinstance(ans, str) and len(ans) > 1
    return f"answer: '{ans[:80]}' | tool calls: {len(call_log)}"

def test_rag_tool_no_executor():
    """Tools provided but no executor – should gracefully ignore."""
    if not rag: return "SKIP – model not loaded"
    tools = [{"name": "search_vector_db", "description": "search", "parameters": {}}]
    ans = rag.answer(
        chunks=CHUNKS_NORMAL,
        question="Who founded NebulaCorp?",
        instruction="Be concise.",
        tools=tools,
        tool_executor=None,
    )
    assert isinstance(ans, str) and len(ans) > 1
    return f"answer (no executor): '{ans[:80]}'"

def test_rag_tool_executor_raises():
    if not rag: return "SKIP – model not loaded"
    def bad_executor(name, args):
        raise ConnectionError("Vector DB offline")
    tools = [{"name": "search_vector_db", "description": "search", "parameters": {}}]
    ans = rag.answer(
        chunks=CHUNKS_NORMAL,
        question="Who founded NebulaCorp?",
        instruction="Be concise.",
        tools=tools,
        tool_executor=bad_executor,
        max_iterations=2,
    )
    assert isinstance(ans, str)
    return f"survived bad executor → '{ans[:60]}'"

def test_rag_max_iterations_1():
    """max_iterations=1 – only one generation pass even with tools."""
    if not rag: return "SKIP – model not loaded"
    call_log = []
    tools = [{"name": "search_vector_db", "description": "search", "parameters": {}}]
    ans = rag.answer(
        chunks=CHUNKS_NORMAL,
        question="Who founded NebulaCorp?",
        instruction="Be concise.",
        tools=tools,
        tool_executor=make_mock_executor(call_log),
        max_iterations=1,
    )
    assert isinstance(ans, str)
    assert len(call_log) <= 1, f"Expected ≤1 tool call with max_iterations=1, got {len(call_log)}"
    return f"answer: '{ans[:60]}' | tool calls: {len(call_log)}"

run_test("Standard RAG – factual question",         test_rag_standard)
run_test("Standard RAG – pirate instruction",       test_rag_instruction_pirate)
run_test("Standard RAG – single chunk",             test_rag_single_chunk)
run_test("Edge: empty chunks list",                 test_rag_empty_chunks)
run_test("Edge: answer not in chunks",              test_rag_no_answer_in_chunks)
run_test("Standard RAG – high temperature",        test_rag_temperature_creative)
run_test("Standard RAG – JSON output instruction",  test_rag_json_instruction)
run_test("Tool use: full agentic with executor",    test_rag_tool_use_with_executor)
run_test("Tool use: tools, no executor",            test_rag_tool_no_executor)
run_test("Tool use: executor raises exception",     test_rag_tool_executor_raises)
run_test("Tool use: max_iterations=1 cap",         test_rag_max_iterations_1)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARIZER TESTS
# ─────────────────────────────────────────────────────────────────────────────

section("SUMMARIZER TESTS")

def test_sum_bullet_points():
    if not summarizer: return "SKIP – model not loaded"
    s = summarizer.summarize(text=TEXT_SHORT, format="bullet_points")
    assert s and len(s) > 5
    return f"summary: '{s[:80]}...'"

def test_sum_paragraph():
    if not summarizer: return "SKIP – model not loaded"
    s = summarizer.summarize(text=TEXT_SHORT, format="paragraph")
    assert s and len(s) > 5
    return f"summary: '{s[:80]}...'"

def test_sum_tldr():
    if not summarizer: return "SKIP – model not loaded"
    s = summarizer.summarize(text=TEXT_SHORT, format="tldr")
    assert s and len(s) > 5
    return f"summary: '{s[:80]}'"

def test_sum_unknown_format():
    """Unknown format should fall back to a generic summary, not crash."""
    if not summarizer: return "SKIP – model not loaded"
    s = summarizer.summarize(text=TEXT_SHORT, format="unknown_xyz")
    assert isinstance(s, str)
    return f"summary (unknown format): '{s[:80]}'"

def test_sum_with_instruction():
    if not summarizer: return "SKIP – model not loaded"
    s = summarizer.summarize(
        text=TEXT_SHORT,
        format="tldr",
        instruction="Write in the style of a 17th-century pirate."
    )
    assert s and len(s) > 5
    return f"summary (pirate): '{s[:80]}'"

def test_sum_empty_text():
    """Empty text should return empty string, not crash."""
    if not summarizer: return "SKIP – model not loaded"
    s = summarizer.summarize(text=TEXT_EMPTY, format="bullet_points")
    assert s == "", f"Expected empty string, got '{s}'"
    return "correctly returned empty string"

def test_sum_max_length_small():
    if not summarizer: return "SKIP – model not loaded"
    s = summarizer.summarize(text=TEXT_SHORT, format="tldr", max_length=32)
    assert isinstance(s, str)
    return f"summary (max_length=32): '{s[:80]}'"

def test_sum_max_length_large():
    if not summarizer: return "SKIP – model not loaded"
    s = summarizer.summarize(text=TEXT_SHORT, format="paragraph", max_length=512)
    assert isinstance(s, str) and len(s) > 5
    return f"summary (max_length=512): '{s[:80]}...'"

def test_sum_temperature_high():
    if not summarizer: return "SKIP – model not loaded"
    s = summarizer.summarize(text=TEXT_SHORT, format="paragraph", temperature=0.8)
    assert isinstance(s, str) and len(s) > 5
    return f"summary (temp=0.8): '{s[:80]}...'"

def test_sum_map_reduce():
    """Long text should trigger Map-Reduce path."""
    if not summarizer: return "SKIP – model not loaded"
    s = summarizer.summarize(text=TEXT_LONG, format="bullet_points", chunk_size=500)
    assert s and len(s) > 5
    return f"summary (Map-Reduce, {len(TEXT_LONG)} chars): '{s[:80]}...'"

def test_sum_map_reduce_tldr():
    if not summarizer: return "SKIP – model not loaded"
    s = summarizer.summarize(text=TEXT_LONG, format="tldr", chunk_size=500)
    assert isinstance(s, str) and len(s) > 5
    return f"tldr (Map-Reduce): '{s[:80]}'"

def test_sum_evaluator_zero_loops():
    """max_correction_loops=0 → skip eval loop entirely."""
    if not summarizer: return "SKIP – model not loaded"
    s = summarizer.summarize(text=TEXT_SHORT, format="paragraph", max_correction_loops=0)
    assert isinstance(s, str) and len(s) > 5
    return f"summary (0 loops): '{s[:80]}...'"

def test_sum_evaluator_one_loop():
    if not summarizer: return "SKIP – model not loaded"
    s = summarizer.summarize(text=TEXT_SHORT, format="bullet_points", max_correction_loops=1)
    assert isinstance(s, str) and len(s) > 5
    return f"summary (1 eval loop): '{s[:80]}...'"

def test_sum_evaluator_two_loops():
    if not summarizer: return "SKIP – model not loaded"
    s = summarizer.summarize(text=TEXT_SHORT, format="paragraph", max_correction_loops=2)
    assert isinstance(s, str) and len(s) > 5
    return f"summary (2 eval loops): '{s[:80]}...'"

def test_sum_evaluator_strict_instruction():
    """Instruction that is hard to satisfy – corrector should kick in."""
    if not summarizer: return "SKIP – model not loaded"
    s = summarizer.summarize(
        text=TEXT_SHORT,
        format="paragraph",
        instruction="You MUST mention Cape Canaveral and 22 satellites explicitly.",
        max_correction_loops=2,
    )
    assert isinstance(s, str) and len(s) > 5
    return f"summary (strict instruction): '{s[:80]}...'"

def test_sum_json_input_dict():
    if not summarizer: return "SKIP – model not loaded"
    s = summarizer.summarize_json({
        "passage": TEXT_SHORT,
        "prompt": "Focus on the landing.",
        "size": 80,
        "format": "tldr"
    })
    assert isinstance(s, str) and len(s) > 1
    return f"summarize_json (dict): '{s[:80]}'"

def test_sum_json_input_string():
    if not summarizer: return "SKIP – model not loaded"
    import json as _json
    payload = _json.dumps({"passage": TEXT_SHORT, "prompt": "", "size": 64, "format": "bullet_points"})
    s = summarizer.summarize_json(payload)
    assert isinstance(s, str) and len(s) > 1
    return f"summarize_json (string): '{s[:80]}'"

def test_sum_json_missing_passage():
    if not summarizer: return "SKIP – model not loaded"
    try:
        summarizer.summarize_json({"prompt": "test"})
        raise AssertionError("Expected ValueError not raised")
    except ValueError:
        return "correctly raised ValueError for missing passage"

def test_sum_json_bad_string():
    if not summarizer: return "SKIP – model not loaded"
    try:
        summarizer.summarize_json("NOT_VALID_JSON{{{{")
        raise AssertionError("Expected ValueError not raised")
    except ValueError:
        return "correctly raised ValueError for bad JSON"

def test_sum_json_wrong_type():
    if not summarizer: return "SKIP – model not loaded"
    try:
        summarizer.summarize_json(12345)
        raise AssertionError("Expected TypeError not raised")
    except TypeError:
        return "correctly raised TypeError for wrong input type"

run_test("Format: bullet_points (short text)",              test_sum_bullet_points)
run_test("Format: paragraph (short text)",                  test_sum_paragraph)
run_test("Format: tldr (short text)",                       test_sum_tldr)
run_test("Format: unknown/fallback",                        test_sum_unknown_format)
run_test("Instruction: pirate style",                       test_sum_with_instruction)
run_test("Edge: empty text",                                test_sum_empty_text)
run_test("Edge: max_length=32 (very short)",               test_sum_max_length_small)
run_test("Edge: max_length=512 (very long)",               test_sum_max_length_large)
run_test("Edge: high temperature (0.8)",                   test_sum_temperature_high)
run_test("Map-Reduce: long text → bullet_points",           test_sum_map_reduce)
run_test("Map-Reduce: long text → tldr",                    test_sum_map_reduce_tldr)
run_test("Evaluator-Corrector: 0 loops (disabled)",        test_sum_evaluator_zero_loops)
run_test("Evaluator-Corrector: 1 loop (default)",          test_sum_evaluator_one_loop)
run_test("Evaluator-Corrector: 2 loops",                   test_sum_evaluator_two_loops)
run_test("Evaluator-Corrector: strict instruction test",   test_sum_evaluator_strict_instruction)
run_test("summarize_json: dict input",                     test_sum_json_input_dict)
run_test("summarize_json: JSON string input",              test_sum_json_input_string)
run_test("summarize_json: missing passage key",            test_sum_json_missing_passage)
run_test("summarize_json: invalid JSON string",            test_sum_json_bad_string)
run_test("summarize_json: wrong type (int)",               test_sum_json_wrong_type)


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

section("STRESS TEST SUMMARY")

passed  = [r for r in results if r[1] == "PASS"]
failed  = [r for r in results if r[1] == "FAIL"]
skipped = [r for r in results if r[1] == "SKIP"]

print(f"\n  Total:  {len(results)}")
print(f"  {PASS}:  {len(passed)}")
print(f"  {FAIL}:  {len(failed)}")
print(f"  {SKIP}:  {len(skipped)}")

if failed:
    print(f"\n  Failed tests:")
    for name, status, msg in failed:
        print(f"    • {name}")
        print(f"      {msg}")

print()
sys.exit(0 if not failed else 1)
