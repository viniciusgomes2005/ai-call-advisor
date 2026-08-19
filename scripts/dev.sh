#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
PYTHON_BIN="${BACKEND_PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" && -x "$ROOT/backend/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/backend/.venv/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Backend virtualenv not found." >&2
  echo 'Run: python3.12 -m venv .venv && .venv/bin/pip install -e "backend[test,asr]"' >&2
  exit 1
fi

if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  echo "Frontend dependencies not found." >&2
  echo "Run: npm --prefix frontend install" >&2
  exit 1
fi

cleanup() {
  trap - INT TERM EXIT
  kill "${BACKEND_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
  wait "${BACKEND_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

(
  cd "$ROOT/backend"
  "$PYTHON_BIN" -m uvicorn app.main:app --reload --host 0.0.0.0 --port "$BACKEND_PORT"
) &
BACKEND_PID=$!

(
  cd "$ROOT/frontend"
  VITE_API_URL="${VITE_API_URL:-http://localhost:$BACKEND_PORT}" npm run dev
) &
FRONTEND_PID=$!

wait -n "$BACKEND_PID" "$FRONTEND_PID"
EXIT_CODE=$?
cleanup
exit "$EXIT_CODE"
