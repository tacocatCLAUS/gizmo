#!/usr/bin/env bash
set -euo pipefail

# Configuration
SYSTEM_FILE="system.txt"
SKILLS_FILE="skills.txt"
MODELFILE="Modelfile"
MODEL_NAME="gizmo"
BASE_MODEL="gemma3:1b"

# Ensure required files exist
for file in "$SYSTEM_FILE" "$SKILLS_FILE"; do
    if [ ! -f "$file" ]; then
        echo "Error: '$file' not found."
        exit 1
    fi
done

# Generate the Modelfile by combining prompts
cat > "$MODELFILE" <<EOF
FROM $BASE_MODEL

# Optional: tweak parameters below:
PARAMETER temperature 0
# PARAMETER num_ctx 32768

SYSTEM """
$(cat "$SYSTEM_FILE")

$(cat "$SKILLS_FILE")
"""
EOF

echo "Generated '$MODELFILE'"

# Build/create the custom model
ollama create "$MODEL_NAME" --file "$MODELFILE"
echo "Model '$MODEL_NAME' created or updated."

# Run the model interactively
echo "Launching '$MODEL_NAME'. Press Ctrl+C to exit."
ollama run "$MODEL_NAME"
