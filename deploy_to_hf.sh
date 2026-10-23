#!/usr/bin/env bash
# ==============================================================================
# NephroScan AI / KidneyAI - Push Deployment to Hugging Face Spaces
# Target Space: https://huggingface.co/spaces/bappa-ai/kidney-stone-detection
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_DIR="$DIR/hf_deployment"
TOKEN="${HF_TOKEN:-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}"

echo "================================================================="
echo "🤗 NEPHROSCAN AI - HUGGING FACE SPACE DEPLOYMENT"
echo "================================================================="
echo "📁 Staging directory: $HF_DIR"

if [ ! -d "$HF_DIR" ]; then
    echo "❌ Error: $HF_DIR does not exist!"
    exit 1
fi

# Ensure latest weights are in place
if [ -f "$DIR/ml_pipeline/models/best_model.h5" ]; then
    mkdir -p "$HF_DIR/models"
    cp -u "$DIR/ml_pipeline/models/best_model.h5" "$HF_DIR/models/best_model.h5"
    echo "✅ Verified latest model checkpoint copied to hf_deployment/models/"
fi

# Ensure sample images exist in hf_deployment/samples
mkdir -p "$HF_DIR/samples"
if [ -f "$DIR/frontend/public/samples/sample_stone.jpg" ]; then
    cp -u "$DIR/frontend/public/samples/sample_stone.jpg" "$HF_DIR/samples/"
fi
if [ -f "$DIR/frontend/public/samples/sample_normal.jpg" ]; then
    cp -u "$DIR/frontend/public/samples/sample_normal.jpg" "$HF_DIR/samples/"
fi

echo "🚀 Uploading directly to Hugging Face Space via HfApi..."
python_bin="$DIR/.venv/bin/python"
if [ ! -f "$python_bin" ]; then
    python_bin="python3"
fi

"$python_bin" - <<EOF
from huggingface_hub import HfApi
import os

token = os.environ.get("HF_TOKEN", "$TOKEN")
api = HfApi(token=token)

print("Authenticating with Hugging Face...")
user = api.whoami()
print(f"Logged in as: {user['name']}")

print("Uploading deployment files to bappa-ai/kidney-stone-detection...")
commit = api.upload_folder(
    folder_path="$HF_DIR",
    repo_id="bappa-ai/kidney-stone-detection",
    repo_type="space",
    ignore_patterns=[".git", ".git/**"],
    commit_message="deploy: automatic update via deploy_to_hf.sh"
)
print(f"🎉 Deployment completed successfully!")
print(f"Commit: {commit}")
EOF

echo ""
echo "================================================================="
echo "✅ Hugging Face Space is live and building:"
echo "👉 https://huggingface.co/spaces/bappa-ai/kidney-stone-detection"
echo "================================================================="
