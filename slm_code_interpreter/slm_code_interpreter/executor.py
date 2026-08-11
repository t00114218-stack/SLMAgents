import subprocess
import sys
import tempfile
import os

def run_code_safely(code: str, timeout: float = 10.0) -> tuple[int, str, str]:
    """
    Executes Python code in a sandboxed subprocess.
    Returns:
        tuple[int, str, str]: (return_code, stdout, stderr)
    """
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        temp_path = f.name
    
    try:
        res = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return res.returncode, res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Execution Timeout Expired."
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
