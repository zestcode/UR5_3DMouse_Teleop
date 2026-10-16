"""INTEGRATION test: the whole dataflow (spacemouse, keyboard, ur5, recorder) under `dora run` with mock mouse,
mock cameras and DRY_RUN=1 (no robot). The first episode starts by itself; Ctrl-C (SIGINT) after ~12 s ends the
run; the session must hold robot.jsonl, two videos whose frame counts equal their CSV rows, and poses."""
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest

import cv2

HERE = Path(__file__).resolve().parents[1]


class TestDataflow(unittest.TestCase):
    def test_mock_run_writes_complete_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "OUT_DIR": tmp, "DRY_RUN": "1", "MOCK_CAMERA": "1", "MOCK_MOUSE": "1",
                   "START_RECORDING": "1"}
            proc = subprocess.Popen(["dora", "run", str(HERE / "dataflow.yml")], cwd=tmp, env=env,
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            time.sleep(12)
            proc.send_signal(signal.SIGINT)
            log = proc.communicate(timeout=30)[0].decode(errors="replace")
            sessions = [p for p in Path(tmp).iterdir() if (p / "session.json").exists()]
            self.assertEqual(len(sessions), 1, log[-3000:])
            session = sessions[0]
            self.assertGreater(sum(1 for _ in (session / "robot.jsonl").open()), 500)  # 125 Hz for ~10 s
            meta = json.loads((session / "session.json").read_text())
            self.assertEqual(meta["episodes"], ["001_episode"])
            info = json.loads((session / "001_episode" / "episode.json").read_text())
            self.assertTrue(info["completed"])
            for role in ("wrist", "fixed"):
                rows = list(csv.DictReader((session / "001_episode" / f"{role}_frames.csv").open()))
                video = cv2.VideoCapture(str(session / "001_episode" / f"{role}.mp4"))
                self.assertGreater(len(rows), 100)
                self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), len(rows))
                self.assertEqual(info["cameras"][role]["frames"], len(rows))
                self.assertEqual(info["cameras"][role]["frame_number_gaps"], 0)
                valid = sum(int(x["pose_valid"]) for x in rows)
                self.assertGreaterEqual(valid, len(rows) - 2)  # the last frames may follow the last robot record


if __name__ == "__main__":
    unittest.main()
