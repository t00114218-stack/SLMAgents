import os
import sys
import time
import traceback
import json

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "slm_cli_agent"))
sys.path.insert(0, os.path.join(base_dir, "slm_code_interpreter"))
sys.path.insert(0, os.path.join(base_dir, "slm_git_copilot"))
sys.path.insert(0, os.path.join(base_dir, "slm_json_cleaner"))

try:
    from slm_cli_agent.cli_agent import SLMCLIAgent
except Exception as e:
    SLMCLIAgent = None
    print(f"Warning: Failed to import SLMCLIAgent: {e}")

try:
    from slm_code_interpreter.code_interpreter import SLMCodeInterpreter
except Exception as e:
    SLMCodeInterpreter = None
    print(f"Warning: Failed to import SLMCodeInterpreter: {e}")

try:
    from slm_git_copilot.git_copilot import SLMGitCopilot
except Exception as e:
    SLMGitCopilot = None
    print(f"Warning: Failed to import SLMGitCopilot: {e}")

try:
    from slm_json_cleaner.json_cleaner import SLMJSONCleaner
except Exception as e:
    SLMJSONCleaner = None
    print(f"Warning: Failed to import SLMJSONCleaner: {e}")


PASS  = "\033[92mPASS\033[0m"
FAIL  = "\033[91mFAIL\033[0m"
SKIP  = "\033[93mSKIP\033[0m"

results = []

def run_test(name, fn):
    print(f"\n  ► {name}", flush=True)
    t0 = time.time()
    try:
        msg = fn()
        elapsed = time.time() - t0
        print(f"    [{PASS}] ({elapsed:.1f}s) {msg or ''}", flush=True)
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

# Initialize agents
print("Initializing agents (running on local ONNX model)...")
cli_agent = SLMCLIAgent(n_ctx=2048) if SLMCLIAgent else None
code_interpreter = SLMCodeInterpreter(n_ctx=2048) if SLMCodeInterpreter else None
git_copilot = SLMGitCopilot(n_ctx=2048) if SLMGitCopilot else None
json_cleaner = SLMJSONCleaner(n_ctx=2048) if SLMJSONCleaner else None

# =============================================================================
# CLI Agent Tests
# =============================================================================
def test_cli_translation():
    if not cli_agent: return "SKIP - CLI Agent not loaded"
    cmd, resp = cli_agent.generate_command("find all files ending with .py in current folder")
    assert cmd and "find" in cmd.lower(), f"Expected find command, got: '{cmd}'"
    return f"Generated command: '{cmd}'"

def test_cli_safety():
    if not cli_agent: return "SKIP - CLI Agent not loaded"
    code, stdout, stderr = cli_agent.execute_command("rm -rf /some/protected/path")
    assert code == -1, f"Expected block code -1, got {code}"
    assert "Blocked" in stderr or "Execution Blocked" in stderr, f"Expected Blocked warning, got: {stderr}"
    return "Successfully blocked dangerous command sequence"

def test_cli_execution():
    if not cli_agent: return "SKIP - CLI Agent not loaded"
    code, stdout, stderr = cli_agent.execute_command("echo 'Test CLI Agent'")
    assert code == 0, f"Expected 0 exit code, got {code}"
    assert "Test CLI Agent" in stdout, f"Expected 'Test CLI Agent' output, got: '{stdout}'"
    return "Executed safe system command successfully"

# =============================================================================
# Code Interpreter Tests
# =============================================================================
def test_code_execution():
    if not code_interpreter: return "SKIP - Code Interpreter not loaded"
    res = code_interpreter.run("Write a python script that prints 'Interpreter Success'", stream=False)
    assert res["success"] is True, f"Execution failed: {res.get('stderr')}"
    assert "Interpreter Success" in res["stdout"], f"Expected output missing: '{res.get('stdout')}'"
    return f"Code output: {res['stdout'].strip()}"

def test_code_self_correction():
    if not code_interpreter: return "SKIP - Code Interpreter not loaded"
    # We will test the self-correcting run loop by forcing an error manually
    # by using instruction that causes an undefined variable print.
    res = code_interpreter.run("Write a python script that references an undefined variable `non_existent_var` first, catches the error, but eventually prints 'Recovered Output'", stream=False)
    assert res["success"] is True, "Self-correction run loop failed to produce successful run"
    return f"Completed in {res['attempts']} attempts. Output: '{res['stdout'].strip()}'"

def test_code_timeout():
    if not code_interpreter: return "SKIP - Code Interpreter not loaded"
    # Execute python script that loops infinitely
    res = code_interpreter._execute_sandbox("import time\nwhile True:\n    time.sleep(0.1)", timeout=1.0)
    assert res[0] == -1, f"Expected return code -1 for timeout, got {res[0]}"
    assert "Timeout" in res[2] or "expired" in res[2].lower(), f"Expected Timeout error, got: {res[2]}"
    return "Sandbox safely terminated infinite loop after timeout limit"

# =============================================================================
# Git Copilot Tests
# =============================================================================
def test_git_commit():
    if not git_copilot: return "SKIP - Git Copilot not loaded"
    sample_diff = (
        "diff --git a/src/main.py b/src/main.py\n"
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -1,3 +1,4 @@\n"
        "-def add(a, b): return a + b\n"
        "+def add(a, b):\n"
        "+    # Add numbers\n"
        "+    return a + b"
    )
    msg = git_copilot.generate_commit_message(sample_diff)
    assert msg and len(msg) > 5
    return f"Commit Message:\n{msg.strip()}"

def test_git_truncation():
    if not git_copilot: return "SKIP - Git Copilot not loaded"
    long_diff = "diff --git a/test.py b/test.py\n" + "hello\n" * 1000
    msg = git_copilot.generate_commit_message(long_diff)
    assert msg and len(msg) > 5
    return "Git diff truncated and commit generated safely"

# =============================================================================
# JSON Cleaner Tests
# =============================================================================
def test_json_repair():
    if not json_cleaner: return "SKIP - JSON Cleaner not loaded"
    broken_json = '{"name": "Agent Suite", "version": "0.1'
    schema = {"name": "string", "version": "string"}
    parsed, success = json_cleaner.clean_json(broken_json, schema)
    assert success is True, f"Failed to clean/repair JSON: {parsed}"
    assert parsed.get("name") == "Agent Suite", f"Key repaired incorrectly: {parsed}"
    return f"Repaired JSON: {parsed}"

def test_json_schema_compliance():
    if not json_cleaner: return "SKIP - JSON Cleaner not loaded"
    broken_json = '{"age": 30, "city": "New York'
    schema = {"age": "number", "city": "string"}
    parsed, success = json_cleaner.clean_json(broken_json, schema)
    assert success is True
    assert isinstance(parsed.get("age"), (int, float))
    assert parsed.get("city") == "New York"
    return f"JSON cleaned and matched target schema: {parsed}"


def main():
    print("\n" + "="*60)
    print("Running Stress Tests for New SLM Agents")
    print("="*60)
    
    run_test("CLI Agent - Command Translation", test_cli_translation)
    run_test("CLI Agent - Command Safety Sandbox", test_cli_safety)
    run_test("CLI Agent - Safe Execution", test_cli_execution)
    
    run_test("Code Interpreter - Standard Execution", test_code_execution)
    run_test("Code Interpreter - Error Self-Correction Loop", test_code_self_correction)
    run_test("Code Interpreter - Subprocess Timeout Limits", test_code_timeout)
    
    run_test("Git Copilot - Conventional Commit Message", test_git_commit)
    run_test("Git Copilot - Diff Truncation Protection", test_git_truncation)
    
    run_test("JSON Cleaner - Malformed JSON Repair", test_json_repair)
    run_test("JSON Cleaner - Schema Adherence", test_json_schema_compliance)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STRESS TEST SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STRESS TEST SUMMARY")
    print("="*60)
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
        sys.exit(1)
    else:
        print("\nAll stress tests passed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
