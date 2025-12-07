#!/usr/bin/env bash
# Helper to export OPENAI_API_KEY for the current shell (interactive use)
# Usage: source scripts/set_openai_key.sh YOUR_KEY

if [ "$1" = "" ]; then
  echo "Usage: source scripts/set_openai_key.sh <OPENAI_API_KEY>"
  return 1 2>/dev/null || exit 1
fi

export OPENAI_API_KEY="$1"
echo "OPENAI_API_KEY set for this shell session."
