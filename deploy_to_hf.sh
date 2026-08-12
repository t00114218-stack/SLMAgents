#!/bin/bash
# deploy_to_hf.sh - Helper script to push SLM Agents monorepo to Hugging Face Spaces
set -e

# Configuration
HF_USERNAME="spcv"
SPACE_NAME="slm-agents"
HF_REMOTE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

echo "=========================================================="
# Get current branch
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "main")
echo "  Deploying branch '$CURRENT_BRANCH' to HF Space: $SPACE_NAME"
echo "=========================================================="
echo ""

# 1. Ensure remote is configured
if git remote | grep -q "^hf$"; then
    echo "[*] Git remote 'hf' is already configured."
else
    echo "[*] Configuring Git remote 'hf'..."
    git remote add hf "$HF_REMOTE_URL"
fi

# 2. Prompt for User Access Token
echo "To push to Hugging Face, you need a User Access Token with WRITE permission."
echo "If you don't have one:"
echo "  1. Go to https://huggingface.co/settings/tokens"
echo "  2. Create a new token with 'write' role."
echo ""
read -rsp "Enter your Hugging Face Write Token: " HF_TOKEN
echo ""
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: Token cannot be empty."
    exit 1
fi

# 3. Commit any unstaged changes to ensure they are deployed
echo ""
echo "[*] Checking for local changes..."
if ! git diff-index --quiet HEAD --; then
    echo "    Found uncommitted changes. Committing them..."
    git add .
    git commit -m "chore: prepare for Hugging Face Spaces deployment"
fi

# 4. Authenticated push to Hugging Face
echo "[*] Pushing to Hugging Face Space repository..."
PUSH_URL="https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

if git push "$PUSH_URL" "${CURRENT_BRANCH}:main" --force; then
    echo ""
    echo "=========================================================="
    echo "🎉 Successfully pushed to Hugging Face!"
    echo "=========================================================="
    echo "Go to your Space page to watch the build progress:"
    echo "  https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
    echo "=========================================================="
else
    echo "❌ Git push failed. Please verify your write token is correct."
    exit 1
fi
