#!/usr/bin/env bash
# General SpaceMouse teleop + recording: RGB video from each D455 + the robot pose (TCP / flange) for every frame.
# Usage: <this dir>/teleop.sh [episode_plan.txt]   line NNN of the plan labels episode NNN; default 001_episode, 002_episode ...
#       first asks: e = continue an existing task, n = new task (its name); episodes go to OUT_DIR/<task>/, numbered
#       from 001 within the task and continuing after its last episode (shown at start-up)
#       DRY_RUN=1 no robot; OUT_DIR=<dir> data location (default <this dir>/data); TASK=<name> skips the question
# Machine settings (robot IP, camera serials, Python env) come from teleop.local.env next to this script
# (copy teleop.local.env.example; not committed). Variables already set in the shell take precedence.
HERE="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
if [ -f "$HERE/teleop.local.env" ]; then
  while IFS= read -r line || [ -n "$line" ]; do  # KEY=value; '#' starts a comment
    line="$(sed 's/#.*//; s/^[[:space:]]*//; s/[[:space:]]*$//' <<<"$line")"
    [ -z "$line" ] && continue
    key="${line%%=*}"
    [ -z "${!key+x}" ] && export "$line"
  done <"$HERE/teleop.local.env"
fi
[ -n "$TELEOP_VENV" ] && export PATH="$TELEOP_VENV/bin:$PATH"
command -v dora >/dev/null || { echo "✗ dora not found: set TELEOP_VENV in $HERE/teleop.local.env (see README)"; exit 1; }
PLAN=""
if [ -n "$1" ]; then PLAN="$(realpath "$1")" || exit 1; fi
export OUT_DIR="$(realpath -m "${OUT_DIR:-$HERE/data}")"
mkdir -p "$OUT_DIR/_logs"
LOG="$OUT_DIR/_logs/teleop_$(date +%Y%m%d_%H%M%S).log"
export DRY_RUN="${DRY_RUN:-0}"
TASK="$(python "$HERE/task.py")" || exit 1
export TASK
echo "Task: $TASK  (episodes in $OUT_DIR/$TASK)"
echo "Preflight:"
python "$HERE/preflight.py" || exit 1
echo "Starting: in 3 s the gripper opens and the robot returns home automatically (keep the e-stop at hand). Full log: $LOG"
trap 'true' INT  # Ctrl-C reaches dora through the keyboard node; this shell only waits
# dora 0.3.8 writes its per-node logs to <this dir>/out/<run id>/
EPISODE_PLAN="$PLAN" dora run "$HERE/dataflow.yml" >"$LOG" 2>&1
status=$?
stty sane 2>/dev/null
if [ "$status" -ne 0 ]; then
  echo
  echo "✗ Exited abnormally ($status). Errors in the log:"
  sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -E "ERROR|Error|Traceback|failed" | tail -15
  echo "Full log: $LOG"
fi
exit "$status"
