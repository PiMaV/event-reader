#!/usr/bin/env bash
# Build EventReader with PyInstaller. From repo root: ./scripts/build.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f EventReader.spec ]; then
  echo "Error: run from the event-reader repo root." >&2
  exit 1
fi
uv sync --group dev
uv run pyinstaller EventReader.spec --noconfirm --clean
if [ -f dist/EventReader.exe ]; then
  echo "Built dist/EventReader.exe"
elif [ -f dist/EventReader ]; then
  echo "Built dist/EventReader"
else
  echo "Error: PyInstaller finished but dist/EventReader was not found." >&2
  exit 1
fi
