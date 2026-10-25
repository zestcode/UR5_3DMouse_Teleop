"""INTEGRATION tests: the whole dataflow (spacemouse, keyboard, ur5, recorder) under `dora run`, and teleop.sh with
its task question, all with mock mouse, mock cameras and DRY_RUN=1 (no robot).
- complete episode: the first episode starts by itself; Ctrl-C (SIGINT) after ~12 s ends the run; the task folder
  must hold the episode (two videos whose frame counts equal their CSV rows, and poses), the run's robot.jsonl,
  task.json with the instruction, and the episode's VPP export (annotation + one 256 x 256 video per camera).
- keys typed into a pseudo-terminal (the dataflow's controlling terminal): Space records, d then y deletes (the
  VPP export too), d then another key keeps; after each episode the robot is reset (gripper open, home); a restart continues the task;
  the (mock) SpaceMouse moves the robot only while an episode is recorded.
- a camera that stops during an episode (MOCK_CAMERA_FAIL): the episode is deleted, the robot reset (gripper open,
  home), the camera restarted, and the next episode takes the number again; a camera that cannot be restarted ends
  the run (dora exits) once the robot is home.
- steps (STEPS): each Enter ends a step, the last one stops the episode; episode.json has the time of each Enter and
  the VPP export the segments between them (segments.json too); Space before the last step saves the episode
  without segments; in a task without steps Enter does not stop it.
- teleop.sh: n + a task name + its instruction + its steps starts that task folder at 001; a restart with e
  continues it without asking the instruction or the steps again."""
import csv
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
import unittest

import cv2
import numpy as np

from tests.ptyutil import PtyProcess

HERE = Path(__file__).resolve().parents[1]


def mock_env(tmp, **extra):
    return {**os.environ, "OUT_DIR": tmp, "DRY_RUN": "1", "MOCK_CAMERA": "1", "MOCK_MOUSE": "1", "TASK": "t", **extra}


def dora(env, cwd, log):
    return PtyProcess(["dora", "run", str(HERE / "dataflow.yml")], env, cwd, stdout=log, stderr=subprocess.STDOUT)


def record(run, name, at):
    run.key(b" ")
    at = run.wait_for(f"Recording {name}", at)
    time.sleep(1.5)
    run.key(b" ")
    return run.wait_for(f"{name} saved", at)


def episode_dirs(task):
    return sorted(p.name for p in task.iterdir() if p.is_dir() and re.match(r"\d+_", p.name))


class TestDataflow(unittest.TestCase):
    def test_mock_run_writes_complete_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = mock_env(tmp, START_RECORDING="1")
            proc = subprocess.Popen(["dora", "run", str(HERE / "dataflow.yml")], cwd=tmp, env=env,
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            time.sleep(12)
            proc.send_signal(signal.SIGINT)
            log = proc.communicate(timeout=30)[0].decode(errors="replace")
            task = Path(tmp) / "t"
            runs = sorted((task / "_runs").glob("*/session.json"))
            self.assertEqual(len(runs), 1, log[-3000:])
            run = runs[0].parent
            with (run / "robot.jsonl").open() as f:
                self.assertGreater(sum(1 for _ in f), 500)  # 125 Hz for ~10 s
            meta = json.loads((run / "session.json").read_text())
            self.assertEqual((meta["task"], meta["run"], meta["episodes"]), ("t", run.name, ["001_episode"]))
            info = json.loads((task / "001_episode" / "episode.json").read_text())
            self.assertTrue(info["completed"])
            self.assertEqual((info["task"], info["run"]), ("t", run.name))
            for role in ("wrist", "fixed"):
                with (task / "001_episode" / f"{role}_frames.csv").open() as f:
                    rows = list(csv.DictReader(f))
                video = cv2.VideoCapture(str(task / "001_episode" / f"{role}.mp4"))
                self.assertGreater(len(rows), 100)
                self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), len(rows))
                self.assertEqual(info["cameras"][role]["frames"], len(rows))
                self.assertEqual(info["cameras"][role]["frame_number_gaps"], 0)
                valid = sum(int(x["pose_valid"]) for x in rows)
                self.assertGreaterEqual(valid, len(rows) - 2)  # the last frames may follow the last robot record
            self.assertEqual((info["instruction"], meta["instruction"]), ("t", "t"))  # default: the task name
            self.assertEqual(json.loads((task / "task.json").read_text()), {"instruction": "t", "steps": []})
            vpp = task / "_vpp"
            ann = json.loads((vpp / "annotation" / "train" / "1.json").read_text())
            self.assertEqual((ann["texts"], ann["episode_id"], ann["cameras"]), (["t"], 1, ["fixed", "wrist"]))
            self.assertEqual((ann["crop"], meta["vpp"]["crop"]), ({"fixed": [150, 100, 500, 450]},) * 2)  # dataflow.yml
            self.assertGreater(ann["video_length"], 80)  # ~10 s at 10 Hz
            n = ann["video_length"]
            self.assertEqual((np.shape(ann["states"]), np.shape(ann["actions"])), ((n, 10), (n, 10)))
            np.testing.assert_allclose(np.diff(ann["source"]["t_ns"]), 1e8)
            for k, role in enumerate(ann["cameras"]):
                video = cv2.VideoCapture(str(vpp / ann["videos"][k]["video_path"]))
                self.assertEqual((int(video.get(cv2.CAP_PROP_FRAME_COUNT)), video.get(cv2.CAP_PROP_FRAME_WIDTH),
                                  video.get(cv2.CAP_PROP_FRAME_HEIGHT)), (n, 256, 256))
                index = ann["source"]["frame_index"][role]
                self.assertTrue(0 <= index[0] and index[-1] < info["cameras"][role]["frames"])
                with (task / "001_episode" / f"{role}_frames.csv").open() as f:
                    t = np.array([int(x["frame_mono_ns"]) for x in csv.DictReader(f)])
                gap_ms = np.diff(t[index[0]:index[-1] + 1]).max() / 1e6
                self.assertLessEqual(ann["source"]["max_offset_ms"][role], gap_ms / 2 + 1e-3)  # the nearest frame
            self.assertIn("VPP: 001_episode exported (train 1,", log)

    def test_d_then_y_deletes_the_last_episode(self):
        with tempfile.TemporaryDirectory() as tmp, open(Path(tmp) / "dora.log", "w") as log:
            run = dora(mock_env(tmp), tmp, log)
            task = Path(tmp) / "t"
            try:
                at = run.wait_for("Ready: press Space")
                meta_file = next((task / "_runs").glob("*/session.json"))
                run.key(b"d")
                at = run.wait_for("No episode of task t to delete", at)
                run.key(b" ")  # answers "no" to the d (nothing asked) and still starts recording
                at = run.wait_for("Recording 001_episode", at)
                run.key(b"d")
                at = run.wait_for("stop it first", at)
                time.sleep(1.5)
                run.key(b" ")  # still stops the recording
                at = run.wait_for("001_episode saved", at)
                self.assertTrue((task / "001_episode").is_dir())
                run.wait_for("VPP: 001_episode exported", at)
                self.assertTrue((task / "_vpp" / "annotation" / "train" / "1.json").is_file())
                run.key(b"d")
                at = run.wait_for("Delete 001_episode? y = delete", at)
                run.key(b"y")
                at = run.wait_for("Deleted 001_episode", at)
                self.assertFalse((task / "001_episode").exists())
                self.assertFalse((task / "_vpp" / "annotation" / "train" / "1.json").exists())
                self.assertFalse((task / "_vpp" / "videos" / "train" / "1").exists())
                self.assertEqual(json.loads(meta_file.read_text())["episodes"], [])
                at = run.wait_for("Next: 001_episode", at)  # the number is free again, once the robot is reset
                self.assertNotIn(b"Wait:", run.out)  # no Space was refused: Next only appears when Space works
                at = record(run, "001_episode", at)
                run.key(b"d")
                at = run.wait_for("Delete 001_episode? y = delete", at)
                run.key(b"n")
                run.wait_for("Kept 001_episode", at)
            finally:
                run.stop()
            self.assertEqual(json.loads(meta_file.read_text())["episodes"], ["001_episode"])
            self.assertEqual(episode_dirs(task), ["001_episode"])
            self.assertTrue(json.loads((task / "001_episode" / "episode.json").read_text())["completed"])

    def test_robot_returns_home_after_each_episode(self):
        with tempfile.TemporaryDirectory() as tmp, open(Path(tmp) / "dora.log", "w") as log:
            run = dora(mock_env(tmp), tmp, log)
            try:
                at = run.wait_for("Ready: press Space")
                run.key(b" ")
                at = run.wait_for("Recording 001_episode", at)
                time.sleep(1.5)
                run.key(b"  ")  # stop, and Space again at once: refused until saved and home
                run.wait_for("Wait: saving the last episode", at)
                at = run.wait_for("001_episode saved", at)
                at = run.wait_for("Reset for the next episode: opening the gripper, returning home", at)
                at = run.wait_for("Next: 002_episode", at)
                self.assertNotIn(b"Recording 002_episode", run.out[:at])
                run.key(b" ")
                run.wait_for("Recording 002_episode", at)
            finally:
                run.stop()
            text = (Path(tmp) / "dora.log").read_text()
            self.assertIn("DRY reset: would moveJ home", text)
            self.assertEqual(text.count("Opening the gripper"), 2)  # start-up + the one reset
            self.assertLess(text.rindex("Opening the gripper"), text.index("DRY reset: would moveJ home"))

    def test_spacemouse_moves_the_robot_only_while_recording(self):
        with tempfile.TemporaryDirectory() as tmp, open(Path(tmp) / "dora.log", "w") as log:
            run = dora(mock_env(tmp, MOCK_AXES="[0.5, 0, 0, 0, 0, 0]"), tmp, log)  # cap pushed along +x all along
            try:
                at = run.wait_for("Ready: press Space")
                time.sleep(1)  # pushed, not recording
                at = record(run, "001_episode", at)
                at = run.wait_for("Next: 002_episode", at)
                time.sleep(1)  # pushed, between episodes
            finally:
                run.stop()
            task = Path(tmp) / "t"
            info = json.loads((task / "001_episode" / "episode.json").read_text())
            run_dir = next((task / "_runs").iterdir())
            with (run_dir / "robot.jsonl").open() as f:
                records = [json.loads(line) for line in f]
            with (task / "001_episode" / "wrist_frames.csv").open() as f:
                frames = list(csv.DictReader(f))
            start, saved = info["start_host_mono_ns"], info["end_host_mono_ns"]
            stop = int(frames[-1]["host_receive_mono_ns"])  # Space; end_host_mono_ns is the save, 0.2 s later
            before = [r for r in records if r["t_ns"] < start]
            during = [r for r in records if start + 1e8 < r["t_ns"] < stop]
            after = [r for r in records if r["t_ns"] > saved + 1e8]
            self.assertGreater(min(len(before), len(during), len(after)), 50)
            x = lambda rs: [r["tcp_pose"][0] for r in rs]
            self.assertEqual(len(set(x(before))), 1)  # before Space: not one step, although pushed
            self.assertTrue(all(r["hard_stop"] and not r["recording"] for r in before))
            self.assertTrue(all(not r["hard_stop"] and r["recording"] for r in during))
            self.assertGreater(x(during)[-1] - x(during)[0], 0.02)  # 0.04 m/s while recording
            self.assertEqual(len(set(x(after))), 1)  # after the episode: locked again
            self.assertTrue(all(r["hard_stop"] and not r["recording"] for r in after))

    def test_enter_ends_each_step_and_the_last_one_stops(self):
        with tempfile.TemporaryDirectory() as tmp, open(Path(tmp) / "dora.log", "w") as log:
            run = dora(mock_env(tmp, STEPS="open pick place"), tmp, log)
            task = Path(tmp) / "t"
            try:
                at = run.wait_for("Ready: press Space")
                run.key(b"\n")
                at = run.wait_for("Enter ends a step only while recording", at)
                run.key(b" ")
                at = run.wait_for("Recording 001_episode: step 1/3 open", at)
                for now in ("2/3 pick", "3/3 place"):
                    time.sleep(1)
                    run.key(b"\n")
                    at = run.wait_for(f"now step {now}", at)
                time.sleep(1)
                run.key(b"\n")  # the end of the last step stops the episode
                at = run.wait_for("001_episode saved (", at)
                at = run.wait_for("3/3 steps)", at)
                run.wait_for("VPP: 001_episode exported", at)  # in a thread: before or after Next
                at = run.wait_for("Next: 002_episode", at)
                run.key(b" ")
                at = run.wait_for("Recording 002_episode", at)
                time.sleep(1)
                run.key(b"\n")
                at = run.wait_for("now step 2/3 pick", at)
                time.sleep(1)
                run.key(b" ")  # Space after one of the three steps: saved, without segments
                at = run.wait_for("002_episode saved (", at)
                at = run.wait_for("1/3 steps: no segments)", at)
                run.wait_for("VPP: 002_episode exported", at)
            finally:
                run.stop()
            steps = ["open", "pick", "place"]
            self.assertEqual(json.loads((task / "task.json").read_text()), {"instruction": "t", "steps": steps})
            first = json.loads((task / "001_episode" / "episode.json").read_text())
            ends = first["step_end_host_mono_ns"]
            self.assertEqual((first["steps"], len(ends)), (steps, 3))
            self.assertTrue(all(d > 0.95e9 for d in np.diff([first["start_host_mono_ns"], *ends])))  # 1 s apart
            for role in ("wrist", "fixed"):
                with (task / "001_episode" / f"{role}_frames.csv").open() as f:
                    last = int(list(csv.DictReader(f))[-1]["host_receive_mono_ns"])
                self.assertLess(abs(last - ends[-1]), 1e8)  # the recording stopped at the last Enter
            ann = json.loads((task / "_vpp" / "annotation" / "train" / "1.json").read_text())
            segs, t = ann["segments"], ann["source"]["t_ns"]
            self.assertEqual([s["primary_action_verb"] for s in segs], steps)
            self.assertEqual((segs[0]["start_frame"], segs[-1]["end_frame"]), (0, ann["video_length"] - 1))
            for a, b in zip(segs, segs[1:]):
                self.assertEqual(b["start_frame"], a["end_frame"] + 1)  # contiguous
            for seg, end in zip(segs[:-1], ends):
                self.assertTrue(t[seg["end_frame"]] <= end < t[seg["end_frame"] + 1])  # cut at the Enter
            self.assertEqual(json.loads((task / "_vpp" / "segments.json").read_text()),
                             {"1": {"all_frames": ann["video_length"], "segments": segs}})
            second = json.loads((task / "002_episode" / "episode.json").read_text())
            self.assertEqual((second["steps"], len(second["step_end_host_mono_ns"])), (steps, 1))
            self.assertNotIn("segments", json.loads((task / "_vpp" / "annotation" / "train" / "2.json").read_text()))

    def test_enter_in_a_task_without_steps_does_not_stop(self):
        with tempfile.TemporaryDirectory() as tmp, open(Path(tmp) / "dora.log", "w") as log:
            run = dora(mock_env(tmp), tmp, log)
            task = Path(tmp) / "t"
            try:
                at = run.wait_for("Ready: press Space")
                run.key(b" ")
                at = run.wait_for("Recording 001_episode", at)
                run.key(b"\n")
                at = run.wait_for("Task t has no steps: Space stops the episode", at)
                time.sleep(1)
                self.assertNotIn(b"001_episode saved", run.out)
                run.key(b" ")
                at = run.wait_for("001_episode saved (", at)
                run.wait_for("VPP: 001_episode exported", at)
            finally:
                run.stop()
            info = json.loads((task / "001_episode" / "episode.json").read_text())
            self.assertEqual((info["steps"], info["step_end_host_mono_ns"]), ([], []))
            self.assertNotIn("segments", json.loads((task / "_vpp" / "annotation" / "train" / "1.json").read_text()))
            self.assertFalse((task / "_vpp" / "segments.json").exists())

    def test_camera_failure_deletes_the_episode_resets_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp, open(Path(tmp) / "dora.log", "w") as log:
            run = dora(mock_env(tmp, MOCK_CAMERA_FAIL="wrist:1.5"), tmp, log)  # no frames 1.5 s into the episode
            task = Path(tmp) / "t"
            try:
                at = run.wait_for("Ready: press Space")
                meta_file = next((task / "_runs").glob("*/session.json"))
                run.key(b" ")
                at = run.wait_for("Recording 001_episode", at)
                at = run.wait_for("✗ wrist camera: no frame for 1 s, 001_episode deleted", at)
                self.assertFalse((task / "001_episode").exists())
                self.assertEqual(json.loads(meta_file.read_text())["episodes"], [])
                at = run.wait_for("Reset for the next episode: opening the gripper, returning home", at)
                run.wait_for("Next: 001_episode", at)  # the number is free again
                at = run.wait_for("wrist camera delivers frames again", at)  # restarted by its thread
                at = record(run, "001_episode", at)
                run.wait_for("Next: 002_episode", at)  # its reset is done
            finally:
                run.stop()
            self.assertEqual(json.loads(meta_file.read_text())["episodes"], ["001_episode"])
            self.assertEqual(episode_dirs(task), ["001_episode"])
            text = (Path(tmp) / "dora.log").read_text()
            self.assertEqual(text.count("Opening the gripper"), 3)  # start-up, the failure, the saved episode
            self.assertEqual(text.count("DRY reset: would moveJ home"), 2)

    def test_camera_that_cannot_be_restarted_ends_the_run_once_home(self):
        with tempfile.TemporaryDirectory() as tmp, open(Path(tmp) / "dora.log", "w") as log:
            run = dora(mock_env(tmp, MOCK_CAMERA_FAIL="wrist:1:dead"), tmp, log)
            task = Path(tmp) / "t"
            try:
                at = run.wait_for("Ready: press Space")
                run.key(b" ")
                at = run.wait_for("Recording 001_episode", at)
                at = run.wait_for("001_episode deleted", at)
                at = run.wait_for("ending the run", at, timeout=60)  # CAMERA_GIVE_UP_S (30 s) after it stopped
                status = run.proc.wait(timeout=60)  # the whole dataflow stops: nothing left hanging
            finally:
                run.stop()
            self.assertNotEqual(status, 0)
            self.assertEqual(episode_dirs(task), [])
            text = (Path(tmp) / "dora.log").read_text()
            self.assertLess(text.index("DRY reset: would moveJ home"), text.index("ending the run"))

    def test_restart_continues_the_task(self):
        with tempfile.TemporaryDirectory() as tmp, open(Path(tmp) / "dora.log", "w") as log:
            task = Path(tmp) / "t"
            first = dora(mock_env(tmp), tmp, log)
            try:
                at = first.wait_for("Task t: no episodes yet")
                at = first.wait_for("Next: 001_episode", at)
                first.wait_for("Ready: press Space", at)
                record(first, "001_episode", at)
            finally:
                first.stop()
            second = dora(mock_env(tmp), tmp, log)
            try:
                at = second.wait_for("Task t: last episode 001_episode (")
                at = second.wait_for("s, saved)", at)
                at = second.wait_for("Next: 002_episode", at)
                second.wait_for("Ready: press Space", at)
                at = record(second, "002_episode", at)
                second.key(b"d")
                at = second.wait_for("Delete 002_episode? y = delete", at)
                second.key(b"y")
                at = second.wait_for("Deleted 002_episode", at)
                at = second.wait_for("Next: 002_episode", at)
                second.key(b"d")  # now the last episode is the one of the first run
                at = second.wait_for("Delete 001_episode? y = delete", at)
                second.key(b"y")
                at = second.wait_for("Deleted 001_episode", at)
                at = second.wait_for("Next: 001_episode", at)
                record(second, "001_episode", at)
            finally:
                second.stop()
            run1, run2 = sorted((task / "_runs").iterdir())
            self.assertEqual(json.loads((run1 / "session.json").read_text())["episodes"], [])
            self.assertEqual(json.loads((run2 / "session.json").read_text())["episodes"], ["001_episode"])
            self.assertEqual(episode_dirs(task), ["001_episode"])
            self.assertEqual(json.loads((task / "001_episode" / "episode.json").read_text())["run"], run2.name)

    def test_teleop_sh_new_task_then_continue_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in mock_env(tmp).items() if k != "TASK"}  # teleop.sh asks for it
            first = PtyProcess([str(HERE / "teleop.sh")], env, tmp)
            try:
                at = first.wait_for("No tasks in")
                at = first.wait_for("Name of the new task: ", at)
                first.key(b"pick cup\n")
                at = first.wait_for("folder name: pick_cup", at)
                at = first.wait_for('Instruction for this task, one English sentence (Enter = "pick cup"): ', at)
                first.key(b"pick up the cup\n")
                at = first.wait_for("Steps of this task, verbs in order", at)
                first.key(b"pick place\n")
                at = first.wait_for("Instruction: pick up the cup", at)
                at = first.wait_for("Steps: pick -> place", at)
                at = first.wait_for("Task pick_cup: no episodes yet", at)
                at = first.wait_for("Ready: press Space", at)
                record(first, "001_episode", at)
                first.key(b"\x03")  # Ctrl-C quits
                first.proc.wait(timeout=60)
            finally:
                first.stop()
            task = Path(tmp) / "pick_cup"
            self.assertEqual(episode_dirs(task), ["001_episode"])
            self.assertEqual(json.loads((task / "task.json").read_text()),
                             {"instruction": "pick up the cup", "steps": ["pick", "place"]})
            self.assertEqual(json.loads((task / "001_episode" / "episode.json").read_text())["instruction"],
                             "pick up the cup")
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir() if p.is_dir()), ["_logs", "pick_cup"])
            second = PtyProcess([str(HERE / "teleop.sh")], env, tmp)
            try:
                at = second.wait_for("1  pick_cup  (1 episode, last 001_episode)")
                second.key(b"e")  # the only task: no number to choose
                at = second.wait_for("Instruction: pick up the cup", at)  # from task.json: not asked again
                at = second.wait_for("Steps: pick -> place", at)  # nor the steps
                at = second.wait_for("Task pick_cup: last episode 001_episode (", at)
                second.wait_for("Next: 002_episode", at)
            finally:
                second.stop()


if __name__ == "__main__":
    unittest.main()
