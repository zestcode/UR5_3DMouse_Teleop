"""Teleop recorder (Dora node): RGB video of each D455 + the robot pose for every video frame.

Cameras: color only (COLOR_W x COLOR_H bgr8 @ FPS), one rs.pipeline per camera, streaming for the whole run.
Episodes are saved per task (TASK, chosen in teleop.sh / task.py) in OUT_DIR/<task>/, numbered from 001 within the
task: the first episode of a run follows the task's highest-numbered episode (shown at start-up); episode NNN gets
line NNN of EPISODE_PLAN as its label.
Space (`record` input) starts / stops an episode; d then y (`delete` input) deletes the last episode of the task
(also one of an earlier run), and the next episode takes its number and plan label again.
Steps (STEPS, else task.json; asked by task.py): a multi-step task (open pick place close: the verbs of its atomic
sub-skills) ends each step with Enter (`step` input: the time the keyboard read the key), the last Enter also stops
the episode; Space stops it before that (saved with fewer step ends: the VPP export gives it no segments).
The SpaceMouse moves the robot only while an episode is being recorded: the `heartbeat` output (sent while every
camera delivers frames) carries `recording`, and the ur5 node freezes the arm and ignores the gripper buttons
whenever it is false (before Space, while saving / resetting, after the last episode) or the heartbeat is stale.
HOME_AFTER_EPISODE=1: once a stopped episode is saved, `reset` puts the robot back in the episode start state
(gripper opened, then moveJ home); Space starts the next episode only after the ur5 state reports that reset done.
A camera that delivers no frame for CAMERA_FAIL_S during an episode fails it: the episode is deleted (the next one
takes its number), the robot is reset as after a saved episode, and the camera's thread restarts its stream (Space
waits until every camera delivers again). A camera still without frames CAMERA_GIVE_UP_S after it stopped ends the
run once the robot is home. Any error of this node stops the whole dataflow (SIGINT to `dora run`, as Esc does),
so no node is left running without the recorder.
The robot (`state` input, every ur5/state record) is kept for the whole run in robot.jsonl (unchanged JSON lines,
raw evidence).
VPP_CAMERAS set: each saved episode is also exported, in a background thread, to the VPP (Video Prediction Policy)
dataset folder <task>/_vpp/ (vpp.py: 10 Hz steps, every camera at each step, state / action, the instruction);
deleting the episode deletes its export too. Not for episodes ended by an error.

OUT_DIR/<task>/
  task.json                       the task's language instruction (INSTRUCTION) and steps (STEPS), asked by task.py
  <NNN>_<label>/episode.json      task, run, instruction, start / end, per camera frame count + frame-number gaps,
                                  steps + step_end_host_mono_ns (the Enter of each ended step, monotonic clock)
  <NNN>_<label>/<role>.mp4        RGB video (mp4v), frame i = row i of <role>_frames.csv
  <NNN>_<label>/<role>_frames.csv per frame: SDK frame number, timestamp + domain, host receive time, the frame
                                  time on the host monotonic clock, and the TCP / flange pose at that time, the
                                  source of the frame time (sdk / receive)
  _runs/<run>/session.json        one teleop run (<run> = its start, YYYYmmdd_HHMMSS): cameras (serial, color
                                  intrinsics), settings, the episodes it recorded
  _runs/<run>/robot.jsonl         every ur5/state record of the run (t_ns = host CLOCK_MONOTONIC)
  _vpp/                           the VPP export (vpp.py)
Frame time: SDK global_time (host system clock, ms) -> monotonic via (time_ns - monotonic_ns) read at receive;
when that is implausible against the host receive time (after it, or more than MAX_FRAME_LATENCY_MS before it:
RealSense global time can run seconds off, seen 3.9 s ahead for a minute), receive time - the camera's median
latency in the episode instead (frame_time_source = receive; timestamp_ms keeps the SDK value).
Pose at frame time: linear (xyz) + slerp (rotation) between the two bracketing robot records; pose_valid=0 when
the frame lies outside the robot records or the bracket is wider than MAX_POSE_GAP_MS.
Poses: base frame, metres, quaternion x y z w (from the ur5 node's rot6d = first two rows of R).
Env: TASK (task folder name), INSTRUCTION (default: task.json, else the task name), STEPS (verbs separated by
spaces, '' = none; default: task.json, else none), CAMERAS (role:serial,...),
COLOR_W/COLOR_H/FPS, OUT_DIR, EPISODE_PLAN (labels, one per line), MAX_POSE_GAP_MS, VPP_CAMERAS (roles in VPP camera
order, '' = no export), VPP_CROP (role:x0,y0,x1,y1 ..., vpp.parse_crops), VPP_FPS, VPP_SIZE, VPP_VAL_EVERY,
CAMERA_FAIL_S, CAMERA_GIVE_UP_S,
MOCK_CAMERA=1 (synthetic frames, tests), MOCK_CAMERA_FAIL=<role>:<s>[:dead] (tests: that mock camera stops <s> s into
the first episode, as a D455 whose frames stop: no frame within 2 s, then an error; dead: it never comes back),
START_RECORDING=1 (first episode starts by itself as soon as Space could start it, i.e. once the robot is ready;
tests)."""
import bisect
import csv
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import threading
import time

import numpy as np
import pyarrow as pa

from task import default_instruction, describe_steps, parse_steps, read_instruction, read_steps

ROWS = ["index", "frame_number", "timestamp_ms", "timestamp_domain", "host_receive_mono_ns", "frame_mono_ns",
        "pose_valid", "pose_gap_ms", "tcp_x", "tcp_y", "tcp_z", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw",
        "flange_x", "flange_y", "flange_z", "flange_qx", "flange_qy", "flange_qz", "flange_qw", "frame_time_source"]
MAX_FRAME_LATENCY_MS = 100  # SDK frame time to host receive: 10-18 ms measured on the D455s at 640 x 480


def retime(frame_ns, receive_ns, max_latency_ns=MAX_FRAME_LATENCY_MS * 1e6):
    """Frame times: the SDK one, unless implausible against the host receive time (after it, or more than
    max_latency_ns before it); those get receive time - the median of the plausible latencies (0 if none).
    -> (times int64, retimed bool per frame)."""
    frame_ns, receive_ns = np.asarray(frame_ns, dtype=np.int64), np.asarray(receive_ns, dtype=np.int64)
    latency = receive_ns - frame_ns
    ok = (latency >= 0) & (latency <= max_latency_ns)
    typical = int(np.median(latency[ok])) if ok.any() else 0
    return np.where(ok, frame_ns, receive_ns - typical), ~ok


def rot6d_to_mat(d6):
    """rot6d = first two ROWS of R (ur5 node / UMI convention); Gram-Schmidt."""
    d6 = np.asarray(d6, dtype=float)
    b1 = d6[..., :3] / np.linalg.norm(d6[..., :3], axis=-1, keepdims=True)
    b2 = d6[..., 3:] - np.sum(b1 * d6[..., 3:], axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack((b1, b2, np.cross(b1, b2)), axis=-2)


def mat_to_quat(R):
    """(..., 3, 3) -> (..., 4) quaternion x y z w, w >= 0."""
    R = np.asarray(R, dtype=float)
    out = np.empty(R.shape[:-2] + (4,))
    for i in np.ndindex(R.shape[:-2]):
        m = R[i]
        t = np.trace(m)
        if t > 0:
            s = 2 * np.sqrt(1 + t)
            q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
        else:
            k = int(np.argmax(np.diag(m)))
            a, b = (k + 1) % 3, (k + 2) % 3
            s = 2 * np.sqrt(1 + m[k, k] - m[a, a] - m[b, b])
            q = [0.] * 4
            q[k], q[a], q[b], q[3] = s / 4, (m[a, k] + m[k, a]) / s, (m[b, k] + m[k, b]) / s, (m[b, a] - m[a, b]) / s
        q = np.array(q)
        out[i] = q * (1 if q[3] >= 0 else -1)
    return out


def slerp(q0, q1, u):
    """Row-wise slerp of unit quaternions (x y z w), u in [0, 1]."""
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot).clip(-1, 1)
    theta = np.arccos(dot)
    small = theta < 1e-6
    s = np.where(small, 1., np.sin(theta))
    w0 = np.where(small, 1 - u, np.sin((1 - u) * theta) / s)
    w1 = np.where(small, u, np.sin(u * theta) / s)
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def interpolate(t_robot, poses9d, t_query, max_gap_ns):
    """-> (valid, gap_ms, xyz (N,3), quat (N,4)); poses9d (M,9) = xyz + rot6d at sorted t_robot (M,)."""
    t_query = np.asarray(t_query, dtype=np.int64)
    n = len(t_query)
    xyz, quat = np.full((n, 3), np.nan), np.full((n, 4), np.nan)
    gap = np.full(n, np.nan)
    if len(t_robot) < 2 or n == 0:
        return np.zeros(n, bool), gap, xyz, quat
    j = np.searchsorted(t_robot, t_query, side="right")
    inside = (j > 0) & (j < len(t_robot))
    j0, j1 = (j - 1).clip(0, len(t_robot) - 2), j.clip(1, len(t_robot) - 1)
    span = (t_robot[j1] - t_robot[j0]).astype(float)
    gap[inside] = span[inside] / 1e6
    valid = inside & (span <= max_gap_ns) & (span > 0)
    if valid.any():
        a, b = j0[valid], j1[valid]
        u = ((t_query[valid] - t_robot[a]) / span[valid])[:, None]
        p = np.asarray(poses9d, dtype=float)
        xyz[valid] = (1 - u) * p[a, :3] + u * p[b, :3]
        quat[valid] = slerp(mat_to_quat(rot6d_to_mat(p[a, 3:])), mat_to_quat(rot6d_to_mat(p[b, 3:])), u)
    return valid, gap, xyz, quat


def safe_label(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_.") or "episode"


def read_plan(path):
    if not path:
        return []
    lines = (line.split("#", 1)[0].strip() for line in Path(path).read_text().splitlines())
    return [safe_label(line) for line in lines if line]


def last_episode(task):
    """-> (number, path) of the highest-numbered episode directory <task>/<NNN>_<label> (of any run), or (0, None)."""
    found = [(int(m.group(1)), p.name) for p in task.glob("*") if (m := re.match(r"(\d+)_", p.name)) and p.is_dir()]
    number, name = max(found, default=(0, None))
    return number, name and task / name


def describe(path):
    """'005_pick (54 s, saved)', from the episode.json of an episode directory."""
    try:
        info = json.loads((path / "episode.json").read_text())
    except (OSError, ValueError):
        info = {}
    if info.get("completed"):
        state = f"{(info['end_host_mono_ns'] - info['start_host_mono_ns']) / 1e9:.0f} s, saved" + (
            ", ended by an error" if info.get("end_reason") == "error" else "")
    else:
        state = "INCOMPLETE: that run ended while it was recording; d then y deletes it"
    return f"{path.name} ({state})"


def run_meta_path(path):
    """session.json of the run that recorded the episode directory `path` (from its episode.json; older layout
    without a run: the session.json next to the episode)."""
    try:
        run = json.loads((path / "episode.json").read_text()).get("run")
    except (OSError, ValueError):
        run = None
    return path.parent / "_runs" / run / "session.json" if run else path.parent / "session.json"


def forget_episode(meta_path, name):
    """Take a deleted episode out of the episode list in the session.json of its (earlier) run."""
    try:
        meta = json.loads(meta_path.read_text())
        meta["episodes"].remove(name)
    except (OSError, ValueError, KeyError):  # no / unreadable session.json, or the episode is not listed
        return
    meta_path.write_text(json.dumps(meta, indent=1, allow_nan=False))


class Console:
    """Operator messages straight to the terminal (teleop.sh sends dora's output to a log file)."""

    def __init__(self):
        try:
            self.out, self.tty = open("/dev/tty", "w"), True
        except OSError:
            self.out, self.tty = sys.stdout, False
        self.lock = threading.Lock()

    def say(self, text):
        with self.lock:
            self.out.write(("\r\033[K" if self.tty else "") + text + "\n")
            self.out.flush()
        if self.tty:
            print(text, flush=True)  # also in the node log

    def status(self, text):
        if self.tty:
            with self.lock:
                self.out.write("\r\033[K" + text)
                self.out.flush()


class Episode:
    """One episode directory; add() runs in the camera threads (one role per thread), close() in the node loop."""

    def __init__(self, task, index, label, cameras, size, fps, run=None, instruction=None, steps=None):
        import cv2
        self.name = f"{index:03d}_{label}"
        self.path = task / self.name
        self.path.mkdir(parents=False, exist_ok=False)
        self.video = {c.role: cv2.VideoWriter(str(self.path / f"{c.role}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                              fps, size) for c in cameras}
        for role, writer in self.video.items():
            if not writer.isOpened():
                raise RuntimeError(f"cannot open video writer for {role}")
        self.rows = {c.role: [] for c in cameras}
        self.info = {"name": self.name, "label": label, "index": index, "task": task.name, "run": run,
                     "instruction": instruction, "start_host_mono_ns": time.monotonic_ns(),
                     "start_host_unix_ns": time.time_ns(), "completed": False}
        if steps is not None:  # teleop (policy rollouts: none)
            self.info.update(steps=list(steps), step_end_host_mono_ns=[])
        self.write_info()

    def write_info(self):
        (self.path / "episode.json").write_text(json.dumps(self.info, indent=1, allow_nan=False))

    def add(self, role, image, row):
        self.video[role].write(image)
        row["index"] = len(self.rows[role])
        self.rows[role].append(row)

    def close(self, reason, robot_t, robot_tcp, robot_flange, max_gap_ns):
        for writer in self.video.values():
            writer.release()
        self.info.update(end_host_mono_ns=time.monotonic_ns(), end_host_unix_ns=time.time_ns(), end_reason=reason,
                         completed=True, cameras={})
        for role, rows in self.rows.items():
            t, retimed = retime([r["frame_mono_ns"] for r in rows], [r["host_receive_mono_ns"] for r in rows])
            for r, v in zip(rows, t.tolist()):
                r["frame_mono_ns"] = v  # the VPP export reads it too
            ok, gap, tcp_xyz, tcp_q = interpolate(robot_t, robot_tcp, t, max_gap_ns)
            _, _, fl_xyz, fl_q = interpolate(robot_t, robot_flange, t, max_gap_ns)
            numbers = [r["frame_number"] for r in rows]
            with (self.path / f"{role}_frames.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(ROWS)
                for k, r in enumerate(rows):
                    w.writerow([r["index"], r["frame_number"], f"{r['timestamp_ms']:.3f}", r["timestamp_domain"],
                                r["host_receive_mono_ns"], r["frame_mono_ns"], int(ok[k]),
                                "" if np.isnan(gap[k]) else f"{gap[k]:.3f}",
                                *("" if np.isnan(v) else f"{v:.7f}" for v in (*tcp_xyz[k], *tcp_q[k], *fl_xyz[k], *fl_q[k])),
                                "receive" if retimed[k] else "sdk"])
            self.info["cameras"][role] = {
                "frames": len(rows), "frames_with_pose": int(ok.sum()),
                "frame_number_gaps": int(sum(b - a - 1 for a, b in zip(numbers, numbers[1:]) if b - a > 1)),
                "frames_retimed": int(retimed.sum()),
                "fps": (len(rows) - 1) / max((t[-1] - t[0]) / 1e9, 1e-9) if len(rows) > 1 else 0.}
        self.write_info()
        return self.info

    def discard(self):
        """Delete the episode (a camera stopped while it was recorded); the cameras are detached already."""
        for writer in self.video.values():
            writer.release()
        shutil.rmtree(self.path)


class Camera:
    """One D455, color stream only, read in its own thread for the whole run. When its frames stop (no frame within
    2 s) the error is kept in `error`; restart=True (recorder): the thread then restarts the stream until it
    delivers again, else (policy) the thread ends."""

    def __init__(self, role, serial, size, fps, mock=False, restart=False):
        self.role, self.serial, self.size, self.fps, self.mock = role, serial, size, fps, mock
        self.lock, self.episode, self.last_ns, self.error = threading.Lock(), None, 0, None
        self.running, self.intrinsics, self.restart = True, None, restart
        fail = os.getenv("MOCK_CAMERA_FAIL", "").split(":") if mock else []
        self.mock_fail = (float(fail[1]), fail[2:] == ["dead"]) if fail[:1] == [role] else None  # (after s, dead)
        self.mock_dead = False
        if not mock:
            import pyrealsense2 as rs
            self.rs, self.pipe, self.config = rs, rs.pipeline(), rs.config()
            self.config.enable_device(serial)
            self.config.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8, fps)
            profile = self.pipe.start(self.config).get_stream(rs.stream.color).as_video_stream_profile()
            i = profile.get_intrinsics()
            self.intrinsics = {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "ppx": i.ppx,
                               "ppy": i.ppy, "model": str(i.model), "coeffs": list(i.coeffs), "source": "SDK"}
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def mock_stops(self):
        """MOCK_CAMERA_FAIL: True once, when the episode this camera records is `after` s old."""
        with self.lock:
            episode = self.episode
        if self.mock_fail is None or episode is None or \
                time.monotonic_ns() - episode.info["start_host_mono_ns"] < self.mock_fail[0] * 1e9:
            return False
        self.mock_dead, self.mock_fail = self.mock_fail[1], None
        return True

    def frames(self):
        if self.mock:
            n = 0
            while self.running:
                time.sleep(1 / self.fps)
                if self.mock_stops():
                    time.sleep(2)  # wait_for_frames(2000)
                    raise RuntimeError("Frame didn't arrive within 2000 (MOCK_CAMERA_FAIL)")
                n += 1
                image = np.full((self.size[1], self.size[0], 3), n % 256, np.uint8)
                yield image, n, time.time_ns() / 1e6, "mock"
        else:
            while self.running:
                f = self.pipe.wait_for_frames(2000).get_color_frame()
                yield (np.asanyarray(f.get_data()), f.get_frame_number(), f.get_timestamp(),
                       str(f.get_frame_timestamp_domain()).split(".")[-1])

    def loop(self):
        attempt = 0  # restarts since the frames stopped
        while self.running:
            try:
                for image, number, stamp_ms, domain in self.frames():
                    mono, unix = time.monotonic_ns(), time.time_ns()
                    self.last_ns, attempt = mono, 0
                    frame_mono = int(round(stamp_ms * 1e6)) - (unix - mono) if domain in ("global_time", "mock") else mono
                    with self.lock:
                        if self.episode is not None:
                            self.episode.add(self.role, image, {
                                "frame_number": int(number), "timestamp_ms": float(stamp_ms), "timestamp_domain": domain,
                                "host_receive_mono_ns": mono, "frame_mono_ns": frame_mono})
            except Exception as exc:  # the heartbeat stops, so the arm freezes; the node loop sees no frames
                self.error = exc
                if not self.restart:
                    return
            while self.running:
                attempt += 1
                print(f"[recorder] {self.role} camera: {self.error!r}; restarting it (attempt {attempt})", flush=True)
                try:
                    self.reopen(attempt)
                    break
                except Exception as exc:
                    self.error = exc
                    time.sleep(2)

    def reopen(self, attempt):
        """Restart the stream after its frames stopped: pipeline stop + start; from the second attempt on, a
        hardware reset of the device first (it comes back on USB after a few seconds)."""
        if self.mock:
            if self.mock_dead:
                raise RuntimeError("the mock camera does not come back (MOCK_CAMERA_FAIL ...:dead)")
            return
        try:
            self.pipe.stop()
        except RuntimeError:  # not streaming any more
            pass
        if attempt > 1:
            for device in self.rs.context().query_devices():
                if device.get_info(self.rs.camera_info.serial_number) == self.serial:
                    device.hardware_reset()
                    time.sleep(5)
        self.pipe.start(self.config)

    def stop(self):
        self.running = False
        self.thread.join(timeout=3)
        if not self.mock:
            try:
                self.pipe.stop()
            except RuntimeError:  # stopped by a failed restart
                pass


def main():
    from dora import Node
    import vpp
    here = Path(__file__).resolve().parent
    mock = os.getenv("MOCK_CAMERA", "0") == "1"
    size = (int(os.getenv("COLOR_W", "640")), int(os.getenv("COLOR_H", "480")))
    fps = int(os.getenv("FPS", "30"))
    max_gap_ns = float(os.getenv("MAX_POSE_GAP_MS", "50")) * 1e6
    plan = read_plan(os.getenv("EPISODE_PLAN", ""))
    home_after = os.getenv("HOME_AFTER_EPISODE", "0") == "1"
    camera_fail_s, give_up_s = float(os.getenv("CAMERA_FAIL_S", "1")), float(os.getenv("CAMERA_GIVE_UP_S", "30"))
    cameras = os.getenv("CAMERAS", "") or ("wrist:mock,fixed:mock" if mock else "")  # role:serial,... (teleop.local.env)
    if not cameras or "<" in cameras:
        raise ValueError("CAMERAS not configured: set it in teleop.local.env (copy teleop.local.env.example)")
    roles = [item.split(":") for item in cameras.split(",")]
    vpp_cameras = [c for c in os.getenv("VPP_CAMERAS", "").split(",") if c]
    if unknown := sorted(set(vpp_cameras) - {role for role, _ in roles}):
        raise ValueError(f"VPP_CAMERAS {unknown}: not camera roles of CAMERAS ({cameras})")
    vpp_crops = vpp.parse_crops(os.getenv("VPP_CROP", ""))
    if unknown := sorted(set(vpp_crops) - {role for role, _ in roles}):
        raise ValueError(f"VPP_CROP {unknown}: not camera roles of CAMERAS ({cameras})")
    vpp_crops = {c: box for c, box in vpp_crops.items() if c in vpp_cameras}
    vpp_fps, vpp_size = int(os.getenv("VPP_FPS", "10")), int(os.getenv("VPP_SIZE", "256"))
    vpp_val_every = int(os.getenv("VPP_VAL_EVERY", "10"))
    root = Path(os.getenv("OUT_DIR", "data"))  # relative: next to this file
    root = root if root.is_absolute() else here / root
    task_name = os.getenv("TASK", "")
    if not task_name or task_name.startswith(("_", ".")) or "/" in task_name:
        raise ValueError(f"TASK {task_name!r}: start with teleop.sh, it asks for the task (or set TASK=<task name>)")
    task = root / task_name
    run = time.strftime("%Y%m%d_%H%M%S") + ("_mock" if mock else "")
    run_dir = task / "_runs" / run
    run_dir.mkdir(parents=True, exist_ok=False)  # also creates a new task's folder
    instruction = os.getenv("INSTRUCTION", "").strip() or read_instruction(task) or default_instruction(task_name)
    steps = parse_steps(os.environ["STEPS"]) if "STEPS" in os.environ else read_steps(task) or []
    (task / "task.json").write_text(json.dumps({"instruction": instruction, "steps": steps}, indent=1))
    vpp_root = task / "_vpp"
    count, last = last_episode(task)  # count = number of the task's last episode, of any run
    console = Console()
    node = Node()
    cams = [Camera(role, serial, size, fps, mock, restart=True) for role, serial in roles]
    meta = {"schema": "ur5-teleop-rgb-pose-v1", "task": task_name, "run": run, "instruction": instruction,
            "steps": steps, "simulated_camera": mock,
            "start_host_mono_ns": time.monotonic_ns(),
            "start_host_unix_ns": time.time_ns(), "color": {"width": size[0], "height": size[1], "fps": fps,
                                                            "format": "bgr8", "video": "mp4v"},
            "cameras": {c.role: {"serial": c.serial, "color_intrinsics": c.intrinsics} for c in cams},
            "max_pose_gap_ms": max_gap_ns / 1e6, "episode_plan": plan, "episodes": [],
            "vpp": {"cameras": vpp_cameras, "crop": vpp_crops, "fps": vpp_fps, "size": vpp_size,
                    "val_every": vpp_val_every, "folder": "_vpp"} if vpp_cameras else None,
            "timing": "frame_mono_ns = SDK global_time (host system clock) mapped to CLOCK_MONOTONIC at receive "
                      f"(implausible, i.e. after the receive or more than {MAX_FRAME_LATENCY_MS} ms before it: receive "
                      "- median latency, frame_time_source = receive); robot t_ns = CLOCK_MONOTONIC; "
                      "pose = interpolation of robot.jsonl at frame_mono_ns",
            "env": {k: os.environ[k] for k in ("TASK", "INSTRUCTION", "STEPS", "CAMERAS", "COLOR_W", "COLOR_H", "FPS", "OUT_DIR",
                                                "EPISODE_PLAN", "DRY_RUN", "MOCK_CAMERA", "MOCK_MOUSE") if k in os.environ}}

    def save_meta():
        (run_dir / "session.json").write_text(json.dumps(meta, indent=1, allow_nan=False))
    save_meta()
    robot_t, robot_tcp, robot_flange, robot_target, robot_grip = [], [], [], [], []
    exports = {}  # episode name -> its VPP export thread
    episode, end_reason, last_status = None, "stopped", 0

    def label(i):
        return plan[i] if i < len(plan) else "episode"

    def say_next():
        if not plan or count < len(plan):
            console.say(f"  Next: {count + 1:03d}_{label(count)}  press Space to start")
        else:
            console.say(f"  All {len(plan)} episodes of the plan are recorded")

    def clock(ns):
        s = (ns - episode.info["start_host_mono_ns"]) / 1e9
        return f"{int(s // 60):02d}:{int(s % 60):02d}"

    def start():
        nonlocal episode
        episode = Episode(task, count + 1, label(count), cams, size, fps, run, instruction, steps)
        for c in cams:
            with c.lock:
                c.episode = episode
        meta["episodes"].append(episode.name)
        save_meta()
        console.say(f"● Recording {episode.name}" + (f": step 1/{len(steps)} {steps[0]} (Enter = end of the step, "
                                                       "the last Enter stops)" if steps else ""))

    def end_step(key_ns, now):
        """Enter while recording: the current step ended at key_ns; the last step also stops the episode."""
        ends = episode.info["step_end_host_mono_ns"]
        ends.append(key_ns)
        episode.write_info()  # kept even if the run ends before the episode is saved
        k = len(ends)
        console.say(f"  Step {k}/{len(steps)} {steps[k - 1]} ended at {clock(key_ns)}"
                    + (f"; now step {k + 1}/{len(steps)} {steps[k]}" if k < len(steps) else ""))
        if k == len(steps):
            pending.append((stop(), "stopped", now + 2e8))
            beat(now)  # the SpaceMouse loses the robot now

    def stop():
        """Detach the episode from the cameras; it is closed 0.2 s later (pending), once the robot records that
        bracket its last frames have arrived."""
        nonlocal episode, count
        for c in cams:
            with c.lock:
                c.episode = None
        done, episode, count = episode, None, count + 1
        return done

    def finish(done, reason):
        info = done.close(reason, np.array(robot_t, dtype=np.int64), np.array(robot_tcp), np.array(robot_flange),
                          max_gap_ns)
        secs = (info["end_host_mono_ns"] - info["start_host_mono_ns"]) / 1e9
        ended = len(info.get("step_end_host_mono_ns", []))
        stepped = (f", {ended}/{len(info['steps'])} steps" + ("" if ended == len(info["steps"]) else ": no segments")
                   if info.get("steps") else "")
        parts = [f"{r} {c['frames']} frames ({c['frames_with_pose']} with pose, {c['frame_number_gaps']} dropped"
                 + (f", {c['frames_retimed']} with an implausible SDK time" if c["frames_retimed"] else "") + ")"
                 for r, c in info["cameras"].items()]
        console.say(f"✓ {done.name} saved ({secs:.0f} s{stepped}): " + ", ".join(parts))
        if vpp_cameras:
            export(done, reason)

    def export(done, reason):
        """VPP export of a saved episode, in a thread (it reads the videos back: seconds); the robot data of its time
        span is copied here, in the node loop, which alone appends to it."""
        times = {c: np.array([row["frame_mono_ns"] for row in done.rows[c]], dtype=np.int64) for c in vpp_cameras}
        if reason == "error" or min(map(len, times.values())) < 2:
            console.say(f"  VPP: {done.name} not exported ({'ended by an error' if reason == 'error' else 'no frames'})")
            return
        lo = bisect.bisect_left(robot_t, min(int(v[0]) for v in times.values()) - 1e9)
        hi = bisect.bisect_right(robot_t, max(int(v[-1]) for v in times.values()) + 1e9)
        robot = vpp.robot_arrays(robot_t[lo:hi], robot_tcp[lo:hi], robot_target[lo:hi], robot_grip[lo:hi])

        def work():
            try:
                ann = vpp.export(vpp_root, done.path, vpp_cameras, times, robot, instruction, vpp_fps, vpp_size,
                                 vpp_val_every, max_gap_ns, source={"task": task_name, "run": run}, crops=vpp_crops,
                                 steps=done.info["steps"], step_ends=done.info["step_end_host_mono_ns"])
                where = ann["videos"][0]["video_path"].split("/")[1]
                console.say(f"  VPP: {done.name} exported ({where} {ann['episode_id']}, {ann['video_length']} frames"
                            + (f", {len(ann['segments'])} segments" if "segments" in ann else "") + ")")
            except Exception as exc:  # the episode itself is saved; python vpp.py <task> exports it again
                console.say(f"✗ VPP export of {done.name} failed: {exc!r}")
        exports[done.name] = threading.Thread(target=work, daemon=True)
        exports[done.name].start()

    pending = []  # (episode, reason, close at monotonic ns): wait for the robot records after the last frame
    resets, homing = 0, None  # reset requests sent; monotonic ns of the one the robot is still homing for

    def reset_robot():
        """The episode start state for the next one: gripper open, then moveJ home; Next once the robot reports it."""
        nonlocal resets, homing
        resets, homing = resets + 1, time.monotonic_ns()
        node.send_output("reset", pa.array([json.dumps({"id": resets})]))
        console.say("  Reset for the next episode: opening the gripper, returning home ...")

    def close(done, reason):
        """Save a stopped episode, then (HOME_AFTER_EPISODE=1) send the robot home for the next one."""
        finish(done, reason)
        if home_after:
            reset_robot()
        else:
            say_next()

    def fail_episode(roles, now):
        """A camera stopped during the episode: delete it (its number is free again) and reset the robot."""
        nonlocal count
        done = stop()
        done.discard()
        meta["episodes"].remove(done.name)
        save_meta()
        count = last_episode(task)[0]
        console.say(f"✗ {'/'.join(roles)} camera: no frame for {camera_fail_s:g} s, {done.name} deleted "
                    "(the camera is being restarted)")
        beat(now)
        reset_robot()  # also with HOME_AFTER_EPISODE=0: the arm stopped mid-task

    down = {}  # camera role -> monotonic ns since when it delivers no frames (it had delivered, or it failed)

    def watch_cameras(now):
        """Every tick: a camera without frames fails the episode being recorded; a camera back is reported; one
        still down after CAMERA_GIVE_UP_S ends the run, once the robot is home."""
        for c in cams:
            if c.role in down and now - c.last_ns < 5e8:
                del down[c.role]
                console.say(f"  {c.role} camera delivers frames again")
            elif c.role not in down and now - c.last_ns >= 5e8 and (c.last_ns or c.error is not None):
                down[c.role] = c.last_ns or now
        if episode is not None and (failed := [c.role for c in cams if now - c.last_ns >= camera_fail_s * 1e9]):
            fail_episode(failed, now)
        lost = [c for c in cams if c.role in down and now - down[c.role] >= give_up_s * 1e9]
        if lost and (homing is None or now - homing > 60e9):  # moveJ home < 60 s
            console.say(f"✗ {'/'.join(c.role for c in lost)} camera: no frames for {give_up_s:g} s, it could not be "
                        "restarted: ending the run (the robot is home)")
            raise RuntimeError("; ".join(f"{c.role} camera: {c.error!r}" for c in lost))

    def beat(now):
        """Heartbeat to the ur5 node while every camera delivers frames; `recording`: the SpaceMouse may move the
        robot (only while an episode is being recorded)."""
        if all(now - c.last_ns < 5e8 for c in cams):
            node.send_output("heartbeat", pa.array([json.dumps({"t_ns": now, "recording": episode is not None})]))

    def busy():
        """-> why Space cannot start an episode now, or None."""
        if pending:
            return "saving the last episode"
        if not robot_t or homing is not None:
            return "the robot is returning home"
        if stale := [c.role for c in cams if time.monotonic_ns() - c.last_ns >= 5e8]:
            return f"the {'/'.join(stale)} camera is not delivering frames"
        return None

    def ask_delete():
        """d: -> the episode directory to delete if y follows (the task's last one, of any run), or None."""
        if episode is not None:
            console.say(f"  Recording {episode.name}: stop it first (Space), then d deletes it")
            return None
        path = last_episode(task)[1]
        if path is None:
            console.say(f"  No episode of task {task_name} to delete")
        return path

    def delete(path):
        nonlocal count
        while pending:  # stopped < 0.2 s ago: close it before removing it
            close(*pending.pop(0)[:2])
        meta_path = run_meta_path(path)  # read before the episode.json is gone
        if (thread := exports.pop(path.name, None)) is not None:
            thread.join()  # it reads the episode's videos
        try:
            shutil.rmtree(path)
        except OSError as exc:
            console.say(f"✗ Could not delete {path.name}: {exc}")
            return
        vpp.remove(vpp_root, vpp.episode_number(path.name))  # also an export of an earlier run
        if path.name in meta["episodes"]:  # recorded in this run
            meta["episodes"].remove(path.name)
            save_meta()
        else:
            forget_episode(meta_path, path.name)
        count = last_episode(task)[0]
        console.say(f"  Deleted {path.name}")
        if homing is None:  # otherwise Next follows once the robot is home
            say_next()

    asked = None  # episode directory waiting for y after d
    console.say("Cameras started. The SpaceMouse moves the robot only while recording. Space = start / stop "
                "recording; " + (f"Enter = end of a step ({describe_steps(steps)}), the last one stops; " if steps
                                 else "") + "d then y = delete the last episode; h = home; Esc = home and quit.")
    console.say(f"Task {task_name}: last episode {describe(last)}" if last else f"Task {task_name}: no episodes yet")
    auto_start = os.getenv("START_RECORDING", "0") == "1"  # tests: as if Space were pressed as soon as it may be
    say_next()
    stop_all = False
    with (run_dir / "robot.jsonl").open("w", buffering=1) as robot_log:
        try:
            for event in node:
                if event["type"] == "STOP":
                    break
                if event["type"] != "INPUT":
                    continue
                now = time.monotonic_ns()
                if event["id"] == "state":
                    text = event["value"][0].as_py()
                    robot_log.write(text + "\n")
                    r = json.loads(text)
                    robot_t.append(r["t_ns"])
                    robot_tcp.append(r["tcp_pose"])
                    robot_flange.append(r["flange_pose"])
                    robot_target.append(r["target_pose"])
                    robot_grip.append(vpp.gripper_values(r))
                    if homing is not None and r.get("reset_done", 0) >= resets:
                        homing, last_status = None, 0  # status line: Ready at the next tick
                        say_next()
                elif event["id"] == "record":
                    if episode is not None:
                        pending.append((stop(), "stopped", now + 2e8))
                    elif why := busy():
                        console.say(f"  Wait: {why}")
                    else:
                        start()
                    beat(now)  # the SpaceMouse takes / loses the robot now, not at the next tick
                elif event["id"] == "step":  # keyboard: Enter, value = the time it read the key
                    if episode is None:
                        console.say("  Enter ends a step only while recording (Space starts an episode)")
                    elif not steps:
                        console.say(f"  Task {task_name} has no steps: Space stops the episode")
                    else:
                        end_step(int(event["value"][0].as_py()), now)
                elif event["id"] == "delete":  # keyboard: d -> "ask"; the next key -> "yes" (y) or "no" (any other)
                    answer, last_status = event["value"][0].as_py(), 0  # show / clear the question at the next tick
                    if answer == "ask":
                        asked = ask_delete()
                    elif asked:
                        if answer == "yes":
                            delete(asked)
                        else:
                            console.say(f"  Kept {asked.name}")
                        asked = None
                elif event["id"] == "tick":
                    watch_cameras(now)
                    if not robot_t and now - meta["start_host_mono_ns"] > 60e9:  # homing (countdown + moveJ) < 60 s
                        raise RuntimeError("No robot state within 60 s: the ur5 node is not running (see the ur5 errors in the log)")
                    if auto_start and episode is None and not busy():
                        auto_start = False
                        start()
                    beat(now)
                    while pending and pending[0][2] <= now:
                        done, reason, _ = pending.pop(0)
                        close(done, reason)
                    if now - last_status >= 1e9:
                        last_status = now
                        stale = [c.role for c in cams if now - c.last_ns >= 5e8]
                        if not robot_t or homing is not None and now - homing < 60e9:  # moveJ home < 60 s
                            robot = "waiting for the robot to reach home ..."
                        elif homing is not None:
                            robot = "the robot did not reach home within 60 s (ur5 node exited? see the log)"
                        elif now - robot_t[-1] > 5e8:
                            robot = "robot state stopped (ur5 node exited? see the log)"
                        else:
                            robot = ""
                        warn = "; ".join(filter(None, [f"{'/'.join(stale)} camera not delivering frames, robot locked" if stale else "", robot]))
                        if episode is not None:
                            k = len(episode.info["step_end_host_mono_ns"])
                            console.status(f"● Recording {episode.name}  {clock(now)}"
                                           + (f"  step {k + 1}/{len(steps)} {steps[k]}" if steps else "")
                                           + (f"  ⚠ {warn}" if warn else ""))
                        else:
                            ask = f"Delete {asked.name}? y = delete, any other key = keep" if asked else ""
                            console.status("  ".join(filter(None, [ask, f"⚠ {warn}" if warn else ""]))
                                           or "Ready: press Space to start recording (the robot follows the "
                                              "SpaceMouse only while recording)")
        except BaseException as exc:
            end_reason, stop_all = "error", isinstance(exc, Exception)  # not on Ctrl-C: dora is stopping already
            raise
        finally:
            if episode is not None:
                pending.append((stop(), end_reason, 0))
            for done, reason, _ in pending:
                finish(done, reason)
            for name, thread in exports.items():
                if thread.is_alive():
                    console.say(f"  VPP: finishing the export of {name} ...")
                    thread.join(timeout=120)
            for c in cams:
                c.stop()
            meta["end_host_mono_ns"] = time.monotonic_ns()
            save_meta()
            console.say(f"Exited. Recorded {len(meta['episodes'])} episode(s) this run, task {task_name} in {task}")
            if stop_all:  # else the other nodes keep `dora run` going (the ur5 node ends with its inputs,
                os.kill(os.getppid(), signal.SIGINT)  # without homing); parent = `dora run`: stops them all


if __name__ == "__main__":
    main()
