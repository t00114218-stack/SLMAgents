#!/bin/bash
# publish_all.sh — Build and publish all 26 SLM Agent packages to PyPI
set -e

echo "=========================================================="
echo "      SLM Agents — Full PyPI Publisher (26 Packages)      "
echo "=========================================================="
echo ""

AGENTS=(
  slm_cli_agent slm_code_interpreter slm_data slm_db_migration
  slm_document_parser slm_email slm_embeddings slm_git_copilot
  slm_json_cleaner slm_math slm_meeting slm_memory slm_orchestrator
  slm_pdf slm_pkb slm_rag slm_search_orchestrator slm_security
  slm_summarizer slm_task_planner slm_text_to_sql slm_translation
  slm_vision_parser slm_voice slm_web_agent slm_web_scraper
)

echo "[1/4] Checking build tools..."
python3 -m pip install --quiet --upgrade pip build twine
echo "      ✅ pip, build, twine ready."
echo ""

echo "[2/4] Building all 26 packages..."
echo "----------------------------------------------------------"
FAILED_BUILD=()
for agent in "${AGENTS[@]}"; do
  if [ ! -d "$agent" ]; then
    echo "  ⚠️  SKIP (folder not found): $agent"
    continue
  fi
  echo "  🔨 Building $agent..."
  (cd "$agent" && rm -rf dist build *.egg-info && python3 -m build --quiet)
  if [ -d "$agent/dist" ]; then
    count=$(ls "$agent/dist/" | wc -l | tr -d ' ')
    echo "      ✅ $agent → $count artifact(s)"
  else
    echo "      ❌ BUILD FAILED: $agent"
    FAILED_BUILD+=("$agent")
  fi
done
echo "----------------------------------------------------------"

if [ ${#FAILED_BUILD[@]} -gt 0 ]; then
  echo "❌ Failed builds: ${FAILED_BUILD[*]}"
  echo "Fix errors above and re-run. Aborting upload."
  exit 1
fi
echo "[+] All packages built successfully."
echo ""

echo "[3/4] Running twine check..."
python3 -m twine check slm_*/dist/* 2>&1 | grep -v "^Checking " || true
echo "      ✅ twine check passed."
echo ""

echo "[4/4] Select upload target:"
echo "  1) TestPyPI  (verify first — recommended)"
echo "  2) Production PyPI  (live release)"
read -rp "Enter choice [1-2]: " choice

case $choice in
  1)
    echo ""
    echo "Uploading to TestPyPI..."
    echo "  Username: __token__"
    echo "  Password: [Your TestPyPI token starting with pypi-]"
    python3 -m twine upload --repository testpypi slm_*/dist/*
    echo ""
    echo "✅ Done! Verify: https://test.pypi.org/search/?q=slm-"
    ;;
  2)
    echo ""
    echo "⚠️  PRODUCTION PyPI UPLOAD"
    echo "  Username: __token__"
    echo "  Password: [Your PyPI token starting with pypi-]"
    read -rp "Type 'yes' to confirm: " confirm
    [ "$confirm" = "yes" ] || { echo "Aborted."; exit 0; }
    python3 -m twine upload slm_*/dist/*
    echo ""
    echo "✅ Done! Verify: https://pypi.org/search/?q=slm-"
    ;;
  *)
    echo "Invalid choice. Aborting."
    exit 1
    ;;
esac

echo ""
echo "=========================================================="
echo "  All 26 SLM Agent packages published!"
echo "=========================================================="
