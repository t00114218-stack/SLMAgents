#!/bin/bash
set -e

echo "=========================================================="
echo "      SLM Agents PyPI Publisher Helper Script             "
echo "=========================================================="
echo ""
echo "This script will rebuild and upload the distributions of:"
echo "  1. slm_orchestrator (v0.1.2)"
echo "  2. slm-rag          (v0.1.2)"
echo "  3. slm-summarizer   (v0.1.2)"
echo "to PyPI using Twine."
echo ""

# Ensure twine is installed
if ! command -v twine &> /dev/null; then
    echo "[!] 'twine' was not found in your PATH."
    echo "Installing/upgrading 'twine' in your user environment..."
    python3 -m pip install --upgrade twine
fi

# Clean and rebuild all packages to ensure the latest 0.1.2 versions are packaged
echo "Building packages..."
echo "----------------------------------------------------------"
echo "[*] Building slm_orchestrator..."
(cd slm_orchestrator && rm -rf dist build *.egg-info && python3 setup.py sdist bdist_wheel > /dev/null)

echo "[*] Building slm_rag..."
(cd slm_rag && rm -rf dist build *.egg-info && python3 setup.py sdist bdist_wheel > /dev/null)

echo "[*] Building slm_summarizer..."
(cd slm_summarizer && rm -rf dist build *.egg-info && python3 setup.py sdist bdist_wheel > /dev/null)
echo "----------------------------------------------------------"
echo "[+] Build complete!"
echo ""

# Confirm build directories exist
if [ ! -d "slm_orchestrator/dist" ] || [ ! -d "slm_rag/dist" ] || [ ! -d "slm_summarizer/dist" ]; then
    echo "[Error] Pre-built distribution files are missing. Build failed."
    exit 1
fi

echo "Which PyPI repository would you like to upload to?"
echo "  1) TestPyPI (Recommended for verifying packages first)"
echo "  2) Production PyPI (Live release)"
read -rp "Enter choice [1-2]: " choice

case $choice in
    1)
        echo ""
        echo "=========================================================="
        echo "Uploading to TestPyPI..."
        echo "Note: When prompted for credentials:"
        echo "  - Username: __token__"
        echo "  - Password: [Your TestPyPI API Token, starting with pypi-]"
        echo "=========================================================="
        echo ""
        python3 -m twine upload --repository testpypi slm_orchestrator/dist/* slm_rag/dist/* slm_summarizer/dist/*
        ;;
    2)
        echo ""
        echo "=========================================================="
        echo "WARNING: UPLOADING TO PRODUCTION PYPI (LIVE RELEASE)      "
        echo "Note: When prompted for credentials:"
        echo "  - Username: __token__"
        echo "  - Password: [Your Production PyPI API Token, starting with pypi-]"
        echo "=========================================================="
        echo ""
        python3 -m twine upload slm_orchestrator/dist/* slm_rag/dist/* slm_summarizer/dist/*
        ;;
    *)
        echo "[Error] Invalid selection. Aborting."
        exit 1
        ;;
esac

echo ""
echo "=========================================================="
echo "Publishing complete!"
echo "=========================================================="
