#!/usr/bin/env bash
# Ask the running OpenJarvis server one question and print the answer.
#
#   scripts/ask.sh "What are my events for tomorrow?"
#   OJ_URL=https://<mac>.<tailnet>.ts.net scripts/ask.sh "When is Priya's birthday?"
#
# The API key is read from OPENJARVIS_API_KEY or ~/.openjarvis/config.toml and
# never printed.
set -u
if [ $# -lt 1 ]; then
  echo 'usage: scripts/ask.sh "your question"' >&2
  exit 2
fi
OJ_URL="${OJ_URL:-http://127.0.0.1:8000}"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3"; fi
fi
KEY="${OPENJARVIS_API_KEY:-$("$PY" - <<'PYEOF'
import os, pathlib, tomllib
path = pathlib.Path(os.environ.get("OPENJARVIS_CONFIG", "~/.openjarvis/config.toml")).expanduser()
try:
    print(tomllib.loads(path.read_text())["server"]["auth"]["api_key"], end="")
except Exception:
    pass
PYEOF
)}"
if [ -z "$KEY" ]; then
  echo "No API key found. Run: jarvis auth create-key" >&2
  exit 1
fi
BODY="$("$PY" -c 'import json,sys; print(json.dumps({"question": sys.argv[1]}))' "$1")"
START=$(date +%s)
RESPONSE="$(curl -sS --max-time 600 -X POST "$OJ_URL/api/ask" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d "$BODY")" || exit 1
printf '%s' "$RESPONSE" | "$PY" -c '
import json, sys
data = json.load(sys.stdin)
if "answer" in data:
    print(data["answer"])
else:
    print("Error:", data.get("detail", data))
    sys.exit(1)
'
echo "($(( $(date +%s) - START ))s)"
