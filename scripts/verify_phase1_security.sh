#!/usr/bin/env bash
# Verify the Phase 1 security boundary against a running OpenJarvis server.
#
# Usage (from the repository root, with `jarvis serve` running in another
# terminal):
#
#     scripts/verify_phase1_security.sh
#
# Optional environment:
#     OJ_URL         server URL               (default http://127.0.0.1:8000)
#     OJ_MODEL       local model to test with (default from config, else qwen3.5:2b)
#     OLLAMA_PORT    Ollama port              (default 11434)
#
# The API key is read from OPENJARVIS_API_KEY or [server.auth] api_key in
# ~/.openjarvis/config.toml. It is never printed. Compatible with the
# bash 3.2 that ships with macOS.

set -u

OJ_URL="${OJ_URL:-http://127.0.0.1:8000}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
CONFIG="${OPENJARVIS_CONFIG:-$HOME/.openjarvis/config.toml}"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3"; fi
fi

PASS=0
FAIL=0
WARN=0
ok()   { PASS=$((PASS + 1)); printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
bad()  { FAIL=$((FAIL + 1)); printf '  \033[31mFAIL\033[0m  %s\n' "$1"; }
warn() { WARN=$((WARN + 1)); printf '  \033[33mWARN\033[0m  %s\n' "$1"; }

toml_get() {  # toml_get <dotted.key>  -> value or empty
  "$PY" - "$CONFIG" "$1" <<'EOF' 2>/dev/null
import sys, tomllib
try:
    data = tomllib.load(open(sys.argv[1], "rb"))
except Exception:
    sys.exit(0)
node = data
for part in sys.argv[2].split("."):
    if not isinstance(node, dict) or part not in node:
        sys.exit(0)
    node = node[part]
print(str(node).lower() if isinstance(node, bool) else node)
EOF
}

# Status code of a request. Extra curl args follow the URL path.
status() {
  local path="$1"; shift
  curl -s -o /dev/null -w '%{http_code}' --max-time 120 "$@" "$OJ_URL$path"
}

API_KEY="${OPENJARVIS_API_KEY:-}"
if [ -z "$API_KEY" ]; then API_KEY="$(toml_get server.auth.api_key)"; fi
MODEL="${OJ_MODEL:-$(toml_get intelligence.default_model)}"
MODEL="${MODEL:-qwen3.5:2b}"
AUTH=(-H "Authorization: Bearer $API_KEY")
JSON=(-H "Content-Type: application/json")

echo "OpenJarvis Phase 1 security check"
echo "  server: $OJ_URL    model: $MODEL    config: $CONFIG"
echo

echo "1. Configuration"
if [ -n "$API_KEY" ] && [ "${#API_KEY}" -ge 32 ]; then
  ok "API key configured (${#API_KEY} characters, not shown)"
else
  bad "No API key of 32+ characters. Run: jarvis auth create-key"
fi
if [ -f "$CONFIG" ]; then
  # GNU stat first: on Linux, `stat -f` means filesystem status and succeeds.
  perms="$(stat -c '%a' "$CONFIG" 2>/dev/null || stat -f '%Lp' "$CONFIG" 2>/dev/null)"
  if [ "$perms" = "600" ]; then ok "config.toml is private (mode 600)"
  else bad "config.toml mode is $perms; run: chmod 600 \"$CONFIG\""; fi
fi
[ "$(toml_get security.local_only_inference)" = "false" ] \
  && bad "[security] local_only_inference is false" \
  || ok "local-only inference not disabled in config"
[ "$(toml_get security.tool_allowlist_enabled)" = "false" ] \
  && bad "[security] tool_allowlist_enabled is false" \
  || ok "tool allowlist not disabled in config"
[ "$(toml_get server.allow_non_loopback_bind)" = "true" ] \
  && bad "[server] allow_non_loopback_bind is true" \
  || ok "non-loopback binding not allowed in config"
[ "$(toml_get server.allow_public_tunnel)" = "true" ] \
  && bad "[server] allow_public_tunnel is true" \
  || ok "public tunnel not allowed in config"
[ "$(toml_get analytics.enabled)" = "true" ] \
  && warn "[analytics] enabled = true in config (blocked at runtime by local-only; set it to false)" \
  || ok "external analytics not enabled in config"
for var in OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY GOOGLE_API_KEY OPENROUTER_API_KEY; do
  eval "val=\${$var:-}"
  [ -n "$val" ] && warn "$var is set in this shell (ignored under local-only; consider removing it)"
done
echo

echo "2. Listening sockets"
port="${OJ_URL##*:}"; port="${port%%/*}"
listeners="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | awk 'NR>1 {print $9}' | sort -u)"
if [ -z "$listeners" ]; then
  bad "Nothing is listening on port $port. Start the server: jarvis serve"
elif echo "$listeners" | grep -qvE '^(127\.0\.0\.1|\[::1\]|localhost):'; then
  bad "Server port $port is listening beyond loopback: $(echo $listeners)"
else
  ok "Server listens on loopback only ($(echo $listeners))"
fi
ollama_listeners="$(lsof -nP -iTCP:"$OLLAMA_PORT" -sTCP:LISTEN 2>/dev/null | awk 'NR>1 {print $9}' | sort -u)"
if [ -z "$ollama_listeners" ]; then
  bad "Ollama is not listening on port $OLLAMA_PORT"
elif echo "$ollama_listeners" | grep -qvE '^(127\.0\.0\.1|\[::1\]|localhost):'; then
  bad "Ollama (which has no authentication) is listening beyond loopback: $(echo $ollama_listeners). Unset OLLAMA_HOST=0.0.0.0"
else
  ok "Ollama listens on loopback only"
fi
echo

echo "3. Authentication"
s="$(status /health)"; [ "$s" = "200" ] && ok "/health reachable" || bad "/health returned $s"
s="$(status /v1/models)"
[ "$s" = "401" ] && ok "No key -> 401 on /v1/models" || bad "No key -> $s on /v1/models (expected 401)"
s="$(status /v1/models -H 'Authorization: Bearer oj_sk_wrong_key_wrong_key_wrong_key_00')"
[ "$s" = "401" ] && ok "Wrong key -> 401" || bad "Wrong key -> $s (expected 401)"
s="$(status /openapi.json)"
[ "$s" = "401" ] && ok "API schema hidden without key" || bad "/openapi.json without key -> $s (expected 401)"
s="$(status /v1/models "${AUTH[@]}")"
[ "$s" = "200" ] && ok "Correct key -> 200 on /v1/models" || bad "Correct key -> $s on /v1/models"
# WebSocket handshake without a key must be refused (no 101 Switching Protocols).
s="$(status /v1/chat/stream -H 'Connection: Upgrade' -H 'Upgrade: websocket' \
      -H 'Sec-WebSocket-Version: 13' -H 'Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==')"
[ "$s" != "101" ] && ok "WebSocket without key refused ($s)" || bad "WebSocket accepted without a key"
echo

echo "4. Local-only inference"
body() { printf '{"model":"%s","stream":%s,"messages":[{"role":"user","content":"Reply with the single word: ready"}]}' "$1" "$2"; }
for cloud in gpt-4o claude-sonnet-4-5 gemini-2.5-pro openrouter/auto; do
  s="$(status /v1/chat/completions "${AUTH[@]}" "${JSON[@]}" -d "$(body "$cloud" false)")"
  [ "$s" = "403" ] && ok "Cloud model $cloud -> 403" || bad "Cloud model $cloud -> $s (expected 403)"
done
s="$(status /v1/chat/completions "${AUTH[@]}" "${JSON[@]}" -d "$(body gpt-4o true)")"
[ "$s" = "403" ] && ok "Cloud model, streaming -> 403" || bad "Cloud model streaming -> $s (expected 403)"
s="$(status /v1/cloud/reload "${AUTH[@]}" "${JSON[@]}" -d '{"keys":{}}')"
[ "$s" = "403" ] && ok "Cloud engine hot-reload refused" || bad "/v1/cloud/reload -> $s (expected 403)"
echo "  ... asking $MODEL a question (CPU inference can take a minute)"
s="$(status /v1/chat/completions "${AUTH[@]}" "${JSON[@]}" -d "$(body "$MODEL" false)")"
[ "$s" = "200" ] && ok "Local model $MODEL answered" || bad "Local model $MODEL -> $s (expected 200)"
echo

echo "5. Public exposure"
if command -v jarvis >/dev/null 2>&1; then
  if jarvis tunnel </dev/null 2>&1 | grep -q "Refusing to start a public tunnel"; then
    ok "jarvis tunnel refuses to create a public URL"
  else
    bad "jarvis tunnel did not refuse"
  fi
else
  warn "jarvis command not on PATH; skipped tunnel check (activate .venv)"
fi
echo

echo "6. Remote access (Phase 3)"
s="$(status /health -H 'Host: evil.example')"
[ "$s" = "421" ] && ok "Unknown Host header refused (DNS-rebinding guard)" \
  || bad "Host evil.example -> $s (expected 421)"
cc="$(curl -s -o /dev/null -D - --max-time 30 "${AUTH[@]}" "$OJ_URL/v1/models" \
      | tr -d '\r' | awk -F': ' 'tolower($1)=="cache-control"{print $2}')"
case "$cc" in *no-store*) ok "API responses are not cacheable ($cc)";;
  *) bad "API Cache-Control is '$cc' (expected no-store)";; esac
if command -v tailscale >/dev/null 2>&1 \
   || [ -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ]; then
  if command -v jarvis >/dev/null 2>&1; then
    if jarvis remote check; then ok "jarvis remote check passed"
    else bad "jarvis remote check failed (see above)"; fi
  else
    warn "jarvis not on PATH; skipped jarvis remote check"
  fi
else
  warn "Tailscale not installed; remote checks skipped"
fi
echo

echo "Result: $PASS passed, $FAIL failed, $WARN warnings"
[ "$FAIL" -eq 0 ]
