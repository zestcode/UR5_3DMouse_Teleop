"""INTEGRATION tests: the policy dataflow (deploy.yml: keyboard, ur5 in CONTROL=policy, policy) under `dora run`, with
mock cameras, DRY_RUN=1 (no robot) and the fake policy server (tests/fake_policy_server.py: each chunk moves +x and
closes the gripper half way), and deploy.sh in front of it.
- a rollout: the first one starts by itself; Ctrl-C (SIGINT) ends the run. The server must have received 256 x 256
  RGB images of both cameras in VPP order with the instruction and a state; the robot must have followed the
  chunks (+x, at most MAX_LINEAR_M_S) and closed the gripper; the run folder must hold session.json (the served
  model), robot.jsonl, and the rollout as a teleop episode (videos = CSV rows) plus policy.jsonl and the images sent.
- keys typed into a pseudo-terminal: Space stops a rollout (the arm holds, then the robot is reset), Space starts
  the next one.
- a server that serves another crop: the policy node refuses to start.
- deploy.sh (VPP_HOST='': no tunnel) checks the server and reaches the operator prompt.
- deploy.sh (VPP_HOST=local, no server answering) starts VPP_REPO/deploy_ur5_server.sh on this machine with
  VPP_PYTHON, waits for it and reaches the operator prompt (a stand-in script that runs the fake server)."""
import csv
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np

from tests.fake_policy_server import MODEL, STEP_M, FakeServer
from tests.ptyutil import PtyProcess

HERE = Path(__file__).resolve().parents[1]
# Stand-in for AtomicVPP's deploy_ur5_server.sh: start runs the fake policy server with $PY on $PORT.
FAKE_SERVER_SH = """#!/bin/bash
cd "$(dirname "$0")"
case "$1" in
  start) echo "$@" > start_args; "$PY" -m tests.fake_policy_server --port "$PORT" > server.log 2>&1 & echo $! > pid ;;
  status) kill -0 "$(cat pid)" 2>/dev/null ;;
esac
"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def mock_env(tmp, port, **extra):
    return {**os.environ, "OUT_DIR": tmp, "DRY_RUN": "1", "MOCK_CAMERA": "1", "TASK": "t",
            "POLICY_ADDR": f"127.0.0.1:{port}", **extra}


class TestDeploy(unittest.TestCase):
    def setUp(self):
        self.server = FakeServer()

    def tearDown(self):
        self.server.close()

    def test_rollout_follows_the_chunks_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.Popen(["dora", "run", str(HERE / "deploy.yml")], cwd=tmp,
                                    env=mock_env(tmp, self.server.port, START_ROLLOUT="1"),
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            time.sleep(12)
            proc.send_signal(signal.SIGINT)
            log = proc.communicate(timeout=30)[0].decode(errors="replace")
            requests = self.server.requests
            self.assertGreater(len(requests), 5, log[-3000:])  # ~0.55 s apart (auto replanning, 1 ms "inference")
            for r in requests:
                self.assertEqual(([i["camera"] for i in r["images"]], r["image_shapes"]),
                                 (["fixed", "wrist"], [[256, 256, 3]] * 2))
                self.assertEqual((r["instruction"], len(r["state"])), ("t", 10))
                self.assertEqual(len(set(r["image_means"][0])), 1)  # the mock frames are grey
            runs = list((Path(tmp) / "t" / "_policy").iterdir())
            self.assertEqual(len(runs), 1)
            run = runs[0]
            meta = json.loads((run / "session.json").read_text())
            self.assertEqual((meta["model"], meta["rollouts"], meta["instruction"]), (MODEL, ["001_rollout"], "t"))
            with (run / "robot.jsonl").open() as f:
                records = [json.loads(line) for line in f]
            following = [r for r in records if r["plan_id"] is not None]
            self.assertGreater(len(following), 500)  # 125 Hz for most of the ~9 s rollout
            x = np.array([r["target_pose"][0] for r in following])
            t = np.array([r["t_ns"] for r in following]) / 1e9
            self.assertGreater(x[-1] - x[0], .2 * STEP_M * 10 * (t[-1] - t[0]))  # moved along +x with the chunks
            speed = np.diff(x) / np.maximum(np.diff(t), 1e-6)
            self.assertLess(np.percentile(speed, 99), .08 * 1.2)  # MAX_LINEAR_M_S (dry run: host timing jitter)
            events = [e for r in records if r["gripper"] for e in r["gripper"]["events"]]
            self.assertIn("close", events)  # the chunks close the gripper from step 5 on
            rollout = run / "001_rollout"
            info = json.loads((rollout / "episode.json").read_text())
            self.assertTrue(info["completed"])
            self.assertEqual((info["task"], info["policy"]["steps"]), ("t", len(requests)))
            for role in ("fixed", "wrist"):
                with (rollout / f"{role}_frames.csv").open() as f:
                    rows = list(csv.DictReader(f))
                video = cv2.VideoCapture(str(rollout / f"{role}.mp4"))
                self.assertGreater(len(rows), 100)
                self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), len(rows))
                sent = cv2.VideoCapture(str(rollout / f"policy_{role}.mp4"))
                self.assertEqual((int(sent.get(cv2.CAP_PROP_FRAME_COUNT)), sent.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                 (len(requests), 256))
            with (rollout / "policy.jsonl").open() as f:
                steps = [json.loads(line) for line in f]
            self.assertEqual([s["step"] for s in steps], list(range(1, len(requests) + 1)))
            self.assertTrue(all(len(s["actions"]) == 10 and len(s["waypoint_t_ns"]) == 10 for s in steps))
            gaps = np.diff([s["t_obs_ns"] for s in steps]) / 1e9
            self.assertTrue((gaps > .4).all() and (gaps < .9).all(), gaps)  # auto: just before the chunk ends

    def test_space_stops_and_starts_rollouts(self):
        with tempfile.TemporaryDirectory() as tmp, open(Path(tmp) / "dora.log", "w") as log:
            run = PtyProcess(["dora", "run", str(HERE / "deploy.yml")], mock_env(tmp, self.server.port), tmp,
                             stdout=log, stderr=subprocess.STDOUT)
            try:
                at = run.wait_for("Ready: press Space")
                self.assertEqual(self.server.requests, [])  # nothing is asked before Space
                run.key(b" ")
                at = run.wait_for("Rollout 001_rollout", at)
                time.sleep(2)
                run.key(b" ")
                at = run.wait_for("001_rollout stopped (Space)", at)
                n = len(self.server.requests)
                at = run.wait_for("001_rollout saved", at)
                at = run.wait_for("Reset: opening the gripper, returning home", at)
                at = run.wait_for("Ready: Space starts the next rollout", at)
                self.assertEqual(len(self.server.requests), n)  # no request between rollouts
                run.key(b" ")
                run.wait_for("Rollout 002_rollout", at)
                time.sleep(1)
            finally:
                run.stop()
            self.assertGreater(n, 1)
            self.assertGreater(len(self.server.requests), n)
            text = (Path(tmp) / "dora.log").read_text()
            self.assertIn("DRY reset: would moveJ home", text)

    def test_refuses_a_model_trained_on_another_crop(self):
        self.server.model = {**MODEL, "crop": {}}
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(["dora", "run", str(HERE / "deploy.yml")], cwd=tmp, env=mock_env(tmp, self.server.port),
                                  stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
            log = (proc.stdout + proc.stderr).decode(errors="replace")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("crop: model {}", log)
            self.assertEqual(self.server.requests, [])
            self.assertFalse((Path(tmp) / "t" / "_policy").exists())

    def test_deploy_sh_checks_the_server_and_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = mock_env(tmp, self.server.port, VPP_HOST="", INSTRUCTION="t")
            run = PtyProcess([str(HERE / "deploy.sh")], env, tmp)
            try:
                at = run.wait_for("✓ policy server at 127.0.0.1:")
                run.wait_for("Ready: press Space", at, timeout=40)
            finally:
                run.stop()

    def test_deploy_sh_starts_the_local_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, port = Path(tmp) / "vpp", free_port()
            repo.mkdir()
            (repo / "deploy_ur5_server.sh").write_text(FAKE_SERVER_SH)
            env = mock_env(tmp, port, VPP_HOST="local", VPP_REPO=str(repo), VPP_PYTHON=sys.executable,
                           VPP_PORT=str(port), INSTRUCTION="t", PYTHONPATH=str(HERE))
            run = PtyProcess([str(HERE / "deploy.sh")], env, tmp)
            try:
                at = run.wait_for("starting it on this machine")
                at = run.wait_for(f"✓ policy server at 127.0.0.1:{port}", at, timeout=40)
                run.wait_for("Ready: press Space", at, timeout=40)
            finally:
                run.stop()
                if (repo / "pid").exists():
                    subprocess.run(["kill", (repo / "pid").read_text().strip()], stderr=subprocess.DEVNULL)
            self.assertEqual((repo / "start_args").read_text().split(), ["start", "ur5_dp/checkpoints/last.pt", "ema"])


if __name__ == "__main__":
    unittest.main()
