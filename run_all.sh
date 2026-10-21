#!/usr/bin/env bash
# NephroScan AI - One-Click Launcher for Frontend, Backend & ML Pipeline

set -e

# Resolve script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python environment
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

# Run the unified Python runner
exec "$PYTHON_BIN" "$SCRIPT_DIR/run_all.py" "$@"
