#!/usr/bin/env bash
# VPP policy on the UR5 (deploy.yml): the policy server on a GPU gets the live camera images + robot state and sends
# back action chunks, which the robot executes. Rollouts are recorded in OUT_DIR/<task>/_policy/<run>/.
# Usage: <this dir>/deploy.sh   asks for the task like teleop.sh (its instruction is the language condition)
#       DRY_RUN=1 no robot; OUT_DIR=<dir> data location (default <this dir>/data); TASK=<name>, INSTRUCTION=<sentence>
# Policy server (teleop.local.env; shell values take precedence):
#   VPP_HOST   where it runs: local (default: this machine's GPU), an ssh host (e.g. pegasus; key login, see README;
#              reached through an SSH tunnel), or '' = start nothing, the server is already reachable at POLICY_ADDR
#   VPP_REPO   the VPP code + model files there (default: local <this dir>/../vpp, ssh /mnt/raptor/kim/AtomicVPP_LIBERO)
#   VPP_PYTHON python of the conda env vpp there (default: local <conda base>/envs/vpp/bin/python, ssh
#              ~/miniconda3/envs/vpp/bin/python)
#   VPP_PORT   port of the server there and of the tunnel here (default 47110)
#   VPP_CKPT, VPP_WEIGHTS   checkpoint (default ur5_dp/checkpoints/last.pt) and weights (default ema) when this
#              script has to start the server; a server already running keeps its own (stop it to change them:
#              [ssh $VPP_HOST] bash $VPP_REPO/deploy_ur5_server.sh stop)
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
VPP_HOST="${VPP_HOST-local}"
if [ "$VPP_HOST" = local ]; then
  VPP_REPO="${VPP_REPO:-$(dirname "$HERE")/vpp}"
  VPP_PYTHON="${VPP_PYTHON:-$(conda info --base 2>/dev/null)/envs/vpp/bin/python}"
  WHERE="this machine"
  on_server() { bash -c "$1"; }
else
  VPP_REPO="${VPP_REPO:-/mnt/raptor/kim/AtomicVPP_LIBERO}"
  WHERE="$VPP_HOST"
  on_server() { ssh -o BatchMode=yes "$VPP_HOST" "$1"; }
fi
VPP_PORT="${VPP_PORT:-47110}"
VPP_CKPT="${VPP_CKPT:-ur5_dp/checkpoints/last.pt}"
VPP_WEIGHTS="${VPP_WEIGHTS:-ema}"
export POLICY_ADDR="${POLICY_ADDR:-127.0.0.1:$VPP_PORT}"
export OUT_DIR="$(realpath -m "${OUT_DIR:-$HERE/data}")"
mkdir -p "$OUT_DIR/_logs"
LOG="$OUT_DIR/_logs/deploy_$(date +%Y%m%d_%H%M%S).log"
export DRY_RUN="${DRY_RUN:-0}"
CHOICE="$(python "$HERE/task.py")" || exit 1  # two lines: task name, instruction
TASK="${CHOICE%%$'\n'*}"
INSTRUCTION="${CHOICE#*$'\n'}"
export TASK INSTRUCTION
echo "Task: $TASK  (rollouts in $OUT_DIR/$TASK/_policy)"
echo "Instruction: $INSTRUCTION"
echo "Preflight:"
python "$HERE/preflight.py" || exit 1
echo "Policy server:"
TUNNEL=""
cleanup() { [ -n "$TUNNEL" ] && kill "$TUNNEL" 2>/dev/null; }
trap cleanup EXIT
if [ -n "$VPP_HOST" ] && [ "$VPP_HOST" != local ]; then
  ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -N \
      -L "${POLICY_ADDR##*:}:127.0.0.1:$VPP_PORT" "$VPP_HOST" &
  TUNNEL=$!
  sleep 1
  kill -0 "$TUNNEL" 2>/dev/null || { echo "  ✗ SSH tunnel to $VPP_HOST failed (port ${POLICY_ADDR##*:} in use here?)"; exit 1; }
fi
python "$HERE/policy.py" --check
status=$?
if [ "$status" -eq 2 ] && [ -n "$VPP_HOST" ]; then
  SERVER="PORT=$VPP_PORT ${VPP_PYTHON:+PY='$VPP_PYTHON' }bash '$VPP_REPO/deploy_ur5_server.sh'"
  echo "  starting it on $WHERE (loading the model takes up to ~5 min) ..."
  on_server "$SERVER start '$VPP_CKPT' '$VPP_WEIGHTS'" || exit 1
  for _ in $(seq 120); do
    sleep 5
    python "$HERE/policy.py" --check >/dev/null 2>&1; status=$?
    [ "$status" -ne 2 ] && break
    on_server "$SERVER status" >/dev/null 2>&1 \
      || { echo "  ✗ the server stopped; its log:"; on_server "tail -20 '$VPP_REPO/logs/ur5_server.log'"; exit 1; }
  done
  python "$HERE/policy.py" --check; status=$?
fi
[ "$status" -eq 0 ] || exit 1
echo "Starting: in 3 s the gripper opens and the robot returns home. Space starts a rollout: the POLICY moves the"
echo "robot; Space again stops it (the arm holds). Keep the emergency stop at hand. Full log: $LOG"
trap 'true' INT  # Ctrl-C reaches dora through the keyboard node; this shell only waits
dora run "$HERE/deploy.yml" >"$LOG" 2>&1
status=$?
stty sane 2>/dev/null
if [ "$status" -ne 0 ]; then
  echo
  echo "✗ Exited abnormally ($status). Errors in the log:"
  sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -E "ERROR|Error|Traceback|failed" | tail -15
  echo "Full log: $LOG"
fi
exit "$status"
