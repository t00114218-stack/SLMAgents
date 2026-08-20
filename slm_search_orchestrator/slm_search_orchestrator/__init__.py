import os
import sys

_curr_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_curr_dir, "..", ".."))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
if os.path.exists(_root_dir):
    for folder in os.listdir(_root_dir):
        folder_path = os.path.join(_root_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("slm_"):
            if folder_path not in sys.path:
                sys.path.insert(0, folder_path)

from .search_orchestrator import SLMSearchOrchestrator

__all__ = ["SLMSearchOrchestrator"]
