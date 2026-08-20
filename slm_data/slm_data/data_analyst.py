import os
import sys
import yaml

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(base_dir, "slm_code_interpreter"))

try:
    from slm_code_interpreter import SLMCodeInterpreter
except ImportError:
    SLMCodeInterpreter = None

try:
    import pandas as pd
except ImportError:
    pd = None

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_DATA_CONFIG"),
        "./config.yaml",
        "../config.yaml",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
    ]
    for path in config_paths:
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return yaml.safe_load(f) or {}, os.path.abspath(path)
            except Exception:
                pass
    return {}, ""

class SLMDataAnalyst:
    """
    Loads local CSV, Parquet, or Excel files. Answers statistical questions, performs calculations,
    and auto-generates data visualization code blocks.
    """
    def __init__(self, model_path=None):
        self.config, _ = load_config()
        self.interpreter = None
        if SLMCodeInterpreter is not None:
            try:
                self.interpreter = SLMCodeInterpreter(model_path=model_path)
            except Exception:
                self.interpreter = None

    def analyze_file(self, file_path: str, query: str, system_prompt: str = None, user_input: str = None) -> dict:
        """
        Parses data schema and generates analysis script execution.
        """
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"Data file not found: {file_path}",
                "script": "",
                "summary": ""
            }

        columns = []
        if pd is not None:
            try:
                df = pd.read_csv(file_path, nrows=5) if file_path.endswith(".csv") else pd.read_excel(file_path, nrows=5)
                columns = list(df.columns)
            except Exception:
                pass

        script = (
            "import pandas as pd\n"
            f"df = pd.read_csv('{file_path}')\n"
            "print('Data summary:')\n"
            "print(df.describe(include='all'))\n"
        )

        exec_stdout = ""
        execution_error = "The code interpreter is unavailable."
        if self.interpreter:
            try:
                res = self.interpreter.run(f"Write a script loading '{file_path}' to answer: {query}", stream=False)
                if isinstance(res, dict) and res.get("success"):
                    exec_stdout = res.get("stdout", "")
                    script = res.get("code", script)
                    execution_error = ""
                else:
                    execution_error = (res or {}).get("stderr", "Data analysis execution failed.") if isinstance(res, dict) else "Data analysis execution failed."
            except Exception as e:
                execution_error = str(e)

        return {
            "success": not execution_error,
            "file": file_path,
            "columns": columns,
            "script": script,
            "summary": exec_stdout,
            "error": execution_error
        }
