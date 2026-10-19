#!/usr/bin/env bash
# SMOKE test: does the tool start at all? Runs ../teleop.sh with mock mouse + mock cameras + DRY_RUN=1
# and only checks that it reaches its first operator prompt within 15 s. Checks nothing about the output.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
OUT_DIR="$TMP" TASK=smoke DRY_RUN=1 MOCK_CAMERA=1 MOCK_MOUSE=1 setsid "$(dirname "$(realpath "$0")")/../teleop.sh" </dev/null >/dev/null 2>&1 &
pid=$!
for _ in $(seq 30); do
  if grep -qs "Cameras started" "$TMP"/_logs/teleop_*.log; then
    kill -INT -- -"$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    echo "SMOKE PASS: teleop.sh started and reached its first prompt"; exit 0
  fi
  sleep .5
done
kill -INT -- -"$pid" 2>/dev/null; wait "$pid" 2>/dev/null
echo "SMOKE FAIL: no prompt within 15 s"; tail -20 "$TMP"/_logs/teleop_*.log 2>/dev/null; exit 1
