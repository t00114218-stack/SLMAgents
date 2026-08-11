import os
import sys
import tempfile

# Add local packages to python path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "slm_document_parser"))
sys.path.insert(0, os.path.join(base_dir, "slm_vision_parser"))
sys.path.insert(0, os.path.join(base_dir, "slm_web_agent"))
sys.path.insert(0, os.path.join(base_dir, "slm_web_scraper"))
sys.path.insert(0, os.path.join(base_dir, "slm_search_orchestrator"))

from slm_document_parser.document_parser import SLMDocumentParser
from slm_vision_parser.vision_parser import SLMVisionParser
from slm_web_agent.web_agent import SLMWebAgent
from slm_web_scraper.web_scraper import SLMWebScraper
from slm_search_orchestrator.search_orchestrator import SLMSearchOrchestrator

PASS  = "\033[92mPASS\033[0m"
FAIL  = "\033[91mFAIL\033[0m"

def run_test(name, fn):
    print(f"  ► {name}", end="... ", flush=True)
    try:
        msg = fn()
        print(f"[{PASS}] {msg or ''}", flush=True)
        return True
    except Exception as e:
        print(f"[{FAIL}] Error: {e}", flush=True)
        return False

# =============================================================================
# Document Parser Tests
# =============================================================================
def test_doc_parser_extraction():
    # Write a temporary text file
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Document Parser Content Check")
        temp_path = f.name
        
    try:
        parser = SLMDocumentParser.__new__(SLMDocumentParser)
        text = parser.extract_text(temp_path)
        assert "Content Check" in text, f"Expected text, got: {text}"
        return "Extracted plain text file successfully"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_doc_parser_regex():
    parser = SLMDocumentParser.__new__(SLMDocumentParser)
    text = "Some prefix ```json\n{\"key\": \"val\"}\n``` some suffix"
    extracted = parser._extract_json(text)
    assert extracted == "{\"key\": \"val\"}", f"Got: {extracted}"
    return "Extracted JSON from markdown block successfully"

# =============================================================================
# Web Scraper Tests
# =============================================================================
def test_scraper_cleaning():
    scraper = SLMWebScraper.__new__(SLMWebScraper)
    html = "<html><head><style>body {color: red;}</style></head><body><nav>nav text</nav><h1>Actual Content</h1></body></html>"
    cleaned = scraper.clean_html(html)
    assert "Actual Content" in cleaned
    assert "body {color: red;}" not in cleaned
    assert "nav text" not in cleaned
    return "Cleaned style and nav tags from raw HTML successfully"

# =============================================================================
# Search Orchestrator Tests
# =============================================================================
def test_search_execution():
    orch = SLMSearchOrchestrator.__new__(SLMSearchOrchestrator)
    results = orch.execute_search("slm agents", max_results=2)
    assert len(results) > 0
    assert "title" in results[0]
    return f"Retrieved search snippets successfully: '{results[0]['title']}'"

# =============================================================================
# Web Agent Tests
# =============================================================================
def test_web_agent_dom_extraction():
    agent = SLMWebAgent.__new__(SLMWebAgent)
    html = '<div><a href="https://test.com/link">My Link</a><button>My Button</button></div>'
    elements = agent._extract_interactive_elements(html)
    assert len(elements) == 2
    assert elements[0]["type"] == "link"
    assert elements[0]["text"] == "My Link"
    return "Extracted interactive links/buttons from DOM successfully"

# =============================================================================
# Main
# =============================================================================
def main():
    print("="*60)
    print("Running Integration Checks for 5 New SLM Agents")
    print("="*60)
    
    success = True
    success &= run_test("Document Parser - Text Extraction", test_doc_parser_extraction)
    success &= run_test("Document Parser - Regex JSON Extraction", test_doc_parser_regex)
    success &= run_test("Web Scraper - HTML Sanitization", test_scraper_cleaning)
    success &= run_test("Search Orchestrator - DuckDuckGo Snippets", test_search_execution)
    success &= run_test("Web Agent - DOM Interactive Extraction", test_web_agent_dom_extraction)
    
    print("="*60)
    if success:
        print("All integration checks completed successfully!")
        sys.exit(0)
    else:
        print("Some integration checks failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
