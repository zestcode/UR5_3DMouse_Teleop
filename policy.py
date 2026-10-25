"""Policy node (Dora; deploy.yml, started by deploy.sh): the VPP policy server (AtomicVPP_LIBERO
step3_deploy_real_ur5.py at POLICY_ADDR: on this machine's GPU, or through deploy.sh's SSH tunnel) drives the robot.

Cameras as the recorder has them (recorder.Camera: color, COLOR_W x COLOR_H @ FPS), the latest frame of each kept.
A policy step: the latest frame of every VPP camera (VPP_CAMERAS order), prepared as the VPP export prepared the
training videos (prepare(): vpp.square, i.e. the VPP_CROP box cut out FIRST, padded to 4:3 if wider, resized to
VPP_SIZE x VPP_SIZE; then BGR -> RGB, the order the training videos decode in), and the state at the first camera's
frame time (TCP xyz + rot6d interpolated between the ur5/state records, gripper position; vpp.trajectory's state),
go to the server; its reply, the action chunk (absolute targets [xyz, rot6d, gripper], one per 1/VPP_FPS s from that
frame time, as the training actions were), goes to the ur5 node as timed waypoints (`waypoints`; CONTROL=policy
there: interpolated, speed-limited, clamped and leashed like the SpaceMouse target).
Replanning: one request at a time. REPLAN_STEPS=0 (auto): the next observation is taken so that its reply is due
just before the chunk's last waypoint (the measured latency: max of the last 5 + 50 ms), so the arm never waits
and each chunk is used as far as possible; N > 0: N steps after the previous observation.
At start-up the server's description of its training data (cameras, crop, size, fps, state layout) must equal
VPP_CAMERAS / VPP_CROP / VPP_SIZE / VPP_FPS / vpp.STATE_LAYOUT, else the node stops: a different crop gives the
model images it never saw.
Space (`record`) starts / stops a rollout; `heartbeat.recording` = a rollout is running, the only time the ur5
node follows the chunks. A rollout also stops at ROLLOUT_MAX_S, on h (`home`), and on a server error or a reply
later than REQUEST_TIMEOUT_S. HOME_AFTER_ROLLOUT=1: then `reset` (gripper open, moveJ home); Space waits for it.
Each rollout is recorded like a teleop episode (recorder.Episode: <role>.mp4, <role>_frames.csv, episode.json),
plus what the policy saw and answered:
  OUT_DIR/<task>/_policy/<run>/session.json        the server's model (checkpoint, weights, val metrics), settings
  OUT_DIR/<task>/_policy/<run>/robot.jsonl         every ur5/state record of the run
  OUT_DIR/<task>/_policy/<run>/<NNN>_rollout/      episode.json, <role>.mp4, <role>_frames.csv (as recorder.py)
                                     policy.jsonl  per policy step: frame times / numbers, state, chunk, timing
                                     policy_<role>.mp4  the images sent (VPP_SIZE, one frame per policy step)
Env: TASK, INSTRUCTION (default: task.json, else the task name), CAMERAS, COLOR_W/H, FPS, OUT_DIR, MAX_POSE_GAP_MS,
VPP_CAMERAS, VPP_CROP, VPP_SIZE, VPP_FPS, POLICY_ADDR (host:port, default 127.0.0.1:47110), REPLAN_STEPS,
ROLLOUT_MAX_S (0 = no limit), REQUEST_TIMEOUT_S, HOME_AFTER_ROLLOUT, MOCK_CAMERA=1 (tests), START_ROLLOUT=1 (the
first rollout starts by itself once the robot is ready; tests).

CLI: python policy.py --check   connect to POLICY_ADDR, print the served model, compare it with deploy.yml's
                                policy settings; exit 0 = ready, 2 = no server answers, 1 = it does not match.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import socket
import struct
import sys
import time

import numpy as np

PROTOCOL = 1  # step3_deploy_real_ur5.py
HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------------ wire (as step3_deploy_real_ur5.py)

def recv_exact(sock, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("the policy server closed the connection")
        got += k
    return bytes(buf)


def send_msg(sock, header, blobs=()):
    """4-byte big-endian header length, JSON header (+ "blobs": their sizes), the blobs back to back."""
    blobs = [memoryview(np.ascontiguousarray(b)).cast("B") for b in blobs]
    head = json.dumps({**header, "blobs": [b.nbytes for b in blobs]}, allow_nan=False).encode()
    sock.sendall(struct.pack("!I", len(head)) + head)
    for b in blobs:
        sock.sendall(b)


def recv_msg(sock):
    (n,) = struct.unpack("!I", recv_exact(sock, 4))
    header = json.loads(recv_exact(sock, n))
    return header, [recv_exact(sock, k) for k in header.pop("blobs", [])]


class Link:
    """One TCP connection to the policy server; one request at a time."""

    def __init__(self, addr, timeout):
        host, _, port = addr.rpartition(":")
        self.sock = socket.create_connection((host or "127.0.0.1", int(port)), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_msg(self.sock, {"type": "hello", "protocol": PROTOCOL})
        reply, _ = recv_msg(self.sock)
        if reply.get("type") != "hello" or reply.get("protocol") != PROTOCOL:
            raise RuntimeError(f"not a policy server of protocol {PROTOCOL}: {reply}")
        self.model = reply["model"]

    def infer(self, request_id, cameras, images, state, instruction):
        """-> the reply header (actions, relative, infer_ms) + sent_ns / reply_ns (monotonic)."""
        sent = time.monotonic_ns()
        send_msg(self.sock, {"type": "infer", "id": request_id, "instruction": instruction,
                             "state": [float(v) for v in state],
                             "images": [{"camera": c, "shape": list(i.shape)} for c, i in zip(cameras, images)]}, images)
        reply, _ = recv_msg(self.sock)
        if reply.get("type") != "actions" or reply.get("id") != request_id:
            raise RuntimeError(f"policy server: {reply.get('error', reply)}")
        return {**reply, "sent_ns": sent, "reply_ns": time.monotonic_ns()}

    def close(self):
        self.sock.close()


# ------------------------------------------------------------------ observation

def prepare(image_bgr, size, crop=None):
    """A camera frame (BGR, as the cameras deliver it) as the policy sees it: vpp.square (crop first, pad to 4:3,
    resize), then RGB, the channel order the training videos decode in."""
    import vpp
    return np.ascontiguousarray(vpp.square(image_bgr, size, crop)[:, :, ::-1])


def robot_state(t_robot, tcp9, grip_pos, t, max_gap_ns):
    """The state (10: TCP xyz + rot6d, gripper position) at time t, as vpp.trajectory computes the training states
    (pose interpolated between the bracketing records, the gripper of the last record), or None outside them."""
    import recorder
    import vpp
    ok, _, xyz, q = recorder.interpolate(np.asarray(t_robot, np.int64), np.asarray(tcp9), [t], max_gap_ns)
    if not ok[0]:
        return None
    grip = grip_pos[max(int(np.searchsorted(t_robot, t, side="right")) - 1, 0)]
    return np.r_[xyz[0], vpp.quat_to_rot6d(q[0]), grip]


def mismatches(model, cameras, crops, size, fps):
    """-> what differs between the served model's training data and these settings ([] = nothing)."""
    import vpp
    ours = {"cameras": list(cameras), "crop": {c: list(b) for c, b in crops.items() if c in cameras}, "size": size,
            "fps": fps, "state_layout": vpp.STATE_LAYOUT}
    return [f"{k}: model {model.get(k)!r}, here {v!r}" for k, v in ours.items() if model.get(k) != v]


def replan_at(t_obs, chunk, period_ns, steps, latency_ns):
    """Monotonic ns at which to take the next observation after one at t_obs (see the module doc)."""
    return t_obs + (steps * period_ns if steps else (chunk - 1) * period_ns - latency_ns)


def node_env(path, node_id):
    """The env block of one node of a dataflow YAML of this directory (KEY: 'value' lines), as {KEY: value}."""
    env, inside, block = {}, False, False
    for line in Path(path).read_text().splitlines():
        if m := re.match(r"\s*- id: (\S+)", line):
            inside, block = m.group(1) == node_id, False
        elif inside and re.match(r"\s+env:\s*$", line):
            block = True
        elif inside and block and (m := re.match(r"\s+([A-Z][A-Z0-9_]*): '(.*?)'", line)):
            env[m.group(1)] = m.group(2)
    return env


class Frames:
    """The cameras' sink (recorder.Camera calls add() in its thread): keeps the latest frame of each camera, and
    passes every frame on to the rollout being recorded (recorder.Episode)."""

    def __init__(self):
        self.latest, self.episode = {}, None
        self.latency = {}  # role -> recent plausible SDK-time-to-receive latencies (ns)

    def add(self, role, image, row):
        if self.episode is not None:
            self.episode.add(role, image, row)  # the raw row: Episode.close retimes the whole rollout
        lat = row["host_receive_mono_ns"] - row["frame_mono_ns"]
        recent = self.latency.setdefault(role, deque(maxlen=100))
        if 0 <= lat <= 100e6:  # recorder.MAX_FRAME_LATENCY_MS: the SDK time is plausible
            recent.append(lat)
        else:  # implausible SDK time (recorder.retime): receive time - the typical latency
            row = {**row, "frame_mono_ns": row["host_receive_mono_ns"] - int(np.median(recent) if recent else 15e6)}
        self.latest[role] = (image.copy(), row)  # the SDK reuses the frame buffer


# ------------------------------------------------------------------------- node

def main():
    from dora import Node
    import pyarrow as pa
    import recorder
    import vpp
    from task import default_instruction, read_instruction
    mock = os.getenv("MOCK_CAMERA", "0") == "1"
    size = (int(os.getenv("COLOR_W", "640")), int(os.getenv("COLOR_H", "480")))
    fps = int(os.getenv("FPS", "30"))
    max_gap_ns = float(os.getenv("MAX_POSE_GAP_MS", "50")) * 1e6
    cameras = os.getenv("CAMERAS", "") or ("wrist:mock,fixed:mock" if mock else "")
    if not cameras or "<" in cameras:
        raise ValueError("CAMERAS not configured: set it in teleop.local.env (copy teleop.local.env.example)")
    roles = [item.split(":") for item in cameras.split(",")]
    vpp_cameras = [c for c in os.getenv("VPP_CAMERAS", "fixed,wrist").split(",") if c]
    if unknown := sorted(set(vpp_cameras) - {role for role, _ in roles}):
        raise ValueError(f"VPP_CAMERAS {unknown}: not camera roles of CAMERAS ({cameras})")
    crops = {c: box for c, box in vpp.parse_crops(os.getenv("VPP_CROP", "")).items() if c in vpp_cameras}
    vpp_size, vpp_fps = int(os.getenv("VPP_SIZE", "256")), int(os.getenv("VPP_FPS", "10"))
    period = round(1e9 / vpp_fps)
    replan_steps = int(os.getenv("REPLAN_STEPS", "0"))
    max_s = float(os.getenv("ROLLOUT_MAX_S", "0"))
    timeout = float(os.getenv("REQUEST_TIMEOUT_S", "3"))
    home_after = os.getenv("HOME_AFTER_ROLLOUT", "0") == "1"
    addr = os.getenv("POLICY_ADDR", "127.0.0.1:47110")
    root = Path(os.getenv("OUT_DIR", "data"))
    root = root if root.is_absolute() else HERE / root
    task_name = os.getenv("TASK", "")
    if not task_name or task_name.startswith(("_", ".")) or "/" in task_name:
        raise ValueError(f"TASK {task_name!r}: start with deploy.sh, it asks for the task (or set TASK=<task name>)")
    task = root / task_name
    instruction = os.getenv("INSTRUCTION", "").strip() or read_instruction(task) or default_instruction(task_name)

    console = recorder.Console()
    link = Link(addr, timeout)
    model = link.model
    if problems := mismatches(model, vpp_cameras, crops, vpp_size, vpp_fps):
        raise RuntimeError("the served model was trained on other observations than deploy.yml prepares: "
                           + "; ".join(problems))
    chunk_len = int(model["action_seq_len"])
    run = time.strftime("%Y%m%d_%H%M%S") + ("_mock" if mock else "")
    run_dir = task / "_policy" / run
    run_dir.mkdir(parents=True, exist_ok=False)
    meta = {"schema": "ur5-vpp-policy-v1", "task": task_name, "run": run, "instruction": instruction, "model": model,
            "policy_addr": addr, "start_host_mono_ns": time.monotonic_ns(), "start_host_unix_ns": time.time_ns(),
            "settings": {"cameras": vpp_cameras, "crop": crops, "size": vpp_size, "fps": vpp_fps,
                         "replan_steps": replan_steps, "rollout_max_s": max_s, "request_timeout_s": timeout,
                         "color": {"width": size[0], "height": size[1], "fps": fps}},
            "rollouts": []}

    def save_meta():
        (run_dir / "session.json").write_text(json.dumps(meta, indent=1, allow_nan=False))
    save_meta()

    node = Node()
    frames = Frames()
    cams = [recorder.Camera(role, serial, size, fps, mock) for role, serial in roles]
    for c in cams:
        with c.lock:
            c.episode = frames
    meta["cameras"] = {c.role: {"serial": c.serial, "color_intrinsics": c.intrinsics} for c in cams}
    save_meta()
    val = model.get("val") or {}
    console.say(f"Policy: {model['checkpoint']} [{model['weights']}] epoch {model['epoch']} (val: "
                f"{val.get('pos_mm', float('nan')):.1f} mm, gripper {val.get('grip_acc', float('nan')):.3f}) "
                f"on {model['device']}")
    console.say(f'Instruction: "{instruction}"' + ("" if instruction in model["instructions"] else
                f"  ⚠ not a training instruction ({model['instructions']})"))
    console.say("Cameras started. Space = start / stop a rollout (the policy moves the robot); h = home; "
                "Esc = home and quit. Keep the emergency stop at hand.")

    robot_t, robot_tcp, robot_flange = [], [], []  # the whole run, for Episode.close
    recent = deque(maxlen=500)  # the last 4 s: (t_ns, tcp_pose, gripper position), for the state
    pool = ThreadPoolExecutor(max_workers=1)
    rollout = None  # the running one: {"episode", "number", "steps", "log", "videos", "start_ns"}
    future, sent_obs = None, None  # the request in flight and its observation
    next_obs, latencies, plan_id, count = 0, deque(maxlen=5), 0, 0
    pending = []  # (episode, reason, close at ns): wait for the robot records after the last frame
    resets, homing = 0, None
    last_status = 0
    auto_start = os.getenv("START_ROLLOUT", "0") == "1"

    def beat(now):
        if all(now - c.last_ns < 5e8 for c in cams):
            node.send_output("heartbeat", pa.array([json.dumps({"t_ns": now, "recording": rollout is not None})]))

    def busy():
        if pending:
            return "saving the last rollout"
        if not robot_t or homing is not None:
            return "the robot is returning home"
        if future is not None:  # its chunk must not reach the next rollout
            return "the last policy request has not been answered yet"
        return None

    def start():
        nonlocal rollout, next_obs, link, count
        if link is None:
            link = Link(addr, timeout)  # reconnect after an error
            if problems := mismatches(link.model, vpp_cameras, crops, vpp_size, vpp_fps):
                raise RuntimeError("the policy server now serves another model: " + "; ".join(problems))
        count += 1
        episode = recorder.Episode(run_dir, count, "rollout", cams, size, fps, run, instruction)
        episode.info["task"] = task_name  # not the run folder it is in
        episode.write_info()
        rollout = {"episode": episode, "number": count, "steps": 0, "start_ns": time.monotonic_ns(),
                   "log": (episode.path / "policy.jsonl").open("w", buffering=1),
                   "videos": {c: vpp.VideoWriter(episode.path / f"policy_{c}.mp4", (vpp_size, vpp_size), 2)
                              for c in vpp_cameras}}
        frames.episode = episode
        next_obs = 0
        meta["rollouts"].append(episode.name)
        save_meta()
        console.say(f"● Rollout {episode.name}: the policy moves the robot (Space stops it)")

    def stop(reason):
        nonlocal rollout
        frames.episode = None
        done, rollout = rollout, None
        done["log"].close()
        for writer in done["videos"].values():
            writer.close()
        info = done["episode"].info
        info["policy"] = {"steps": done["steps"], "end_reason": reason}
        pending.append((done["episode"], reason, time.monotonic_ns() + 2e8))
        console.say(f"■ {done['episode'].name} stopped ({reason}) after {done['steps']} policy steps, "
                    f"{(time.monotonic_ns() - done['start_ns']) / 1e9:.0f} s")

    def save(episode, reason):
        episode.close("error" if reason == "error" else "stopped", np.array(robot_t, np.int64), np.array(robot_tcp),
                      np.array(robot_flange), max_gap_ns)  # keeps info["policy"]

    def close(episode, reason):
        nonlocal resets, homing
        save(episode, reason)
        console.say(f"✓ {episode.name} saved")
        if home_after:
            resets, homing = resets + 1, time.monotonic_ns()
            node.send_output("reset", pa.array([json.dumps({"id": resets})]))
            console.say("  Reset: opening the gripper, returning home ...")
        else:
            console.say("  Ready: Space starts the next rollout")

    def observe(now):
        """-> the observation of a policy step, or None (a camera or the robot record not there yet)."""
        latest = [frames.latest.get(c) for c in vpp_cameras]
        if any(x is None or now - x[1]["host_receive_mono_ns"] > 2e8 for x in latest):
            return None
        t_obs = latest[0][1]["frame_mono_ns"]
        t, tcp, pos = zip(*recent) if recent else ((), (), ())
        state = robot_state(np.asarray(t, np.int64), tcp, pos, t_obs, max_gap_ns) if len(t) > 1 else None
        if state is None:
            return None
        images = [prepare(image, vpp_size, crops.get(c)) for c, (image, _) in zip(vpp_cameras, latest)]
        return {"t_obs": t_obs, "state": state, "images": images,
                "frame_ns": {c: x[1]["frame_mono_ns"] for c, x in zip(vpp_cameras, latest)},
                "frame_number": {c: x[1]["frame_number"] for c, x in zip(vpp_cameras, latest)}}

    def answered(obs, reply):
        """Chunk -> waypoints for the ur5 node; log the step; schedule the next observation."""
        nonlocal next_obs, plan_id
        actions = np.asarray(reply["actions"], dtype=float)
        t = obs["t_obs"] + np.arange(len(actions), dtype=np.int64) * period
        plan_id += 1
        node.send_output("waypoints", pa.array([json.dumps({
            "id": plan_id, "t_ns": t.tolist(), "poses": actions[:, :9].tolist(), "gripper": actions[:, 9].tolist()})]))
        latencies.append(reply["reply_ns"] - obs["t_obs"])
        next_obs = replan_at(obs["t_obs"], len(actions), period, replan_steps, max(latencies) + 50e6)
        rollout["steps"] += 1
        rollout["log"].write(json.dumps({
            "step": rollout["steps"], "plan_id": plan_id, "t_obs_ns": obs["t_obs"], "frame_ns": obs["frame_ns"],
            "frame_number": obs["frame_number"], "state": obs["state"].round(7).tolist(),
            "sent_ns": reply["sent_ns"], "reply_ns": reply["reply_ns"], "infer_ms": round(reply["infer_ms"], 2),
            "latency_ms": round((reply["reply_ns"] - obs["t_obs"]) / 1e6, 2), "waypoint_t_ns": t.tolist(),
            "actions": actions.round(7).tolist(), "relative": np.round(reply["relative"], 7).tolist()},
            allow_nan=False) + "\n")
        for c, image in zip(vpp_cameras, obs["images"]):
            rollout["videos"][c].write(np.ascontiguousarray(image[:, :, ::-1]))

    with (run_dir / "robot.jsonl").open("w", buffering=1) as robot_log:
        end_reason = "stopped"
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
                    recent.append((r["t_ns"], r["tcp_pose"], vpp.gripper_values(r)[0]))
                    if homing is not None and r.get("reset_done", 0) >= resets:
                        homing, last_status = None, 0
                        console.say("  Ready: Space starts the next rollout")
                elif event["id"] == "record":
                    if rollout is not None:
                        stop("Space")
                    elif why := busy():
                        console.say(f"  Wait: {why}")
                    else:
                        start()
                    beat(now)
                elif event["id"] == "home" and rollout is not None:
                    stop("h")  # the ur5 node homes by itself
                    beat(now)
                elif event["id"] == "tick":
                    for c in cams:
                        if c.error is not None:
                            raise RuntimeError(f"{c.role} camera: {c.error}")
                    if auto_start and rollout is None and not busy():
                        auto_start = False
                        start()
                    if future is not None and future.done():
                        obs, done_future, future = sent_obs, future, None
                        try:
                            reply = done_future.result()
                        except Exception as exc:  # timeout, connection lost, server error: stop moving
                            console.say(f"✗ Policy server: {exc!r}")
                            if rollout is not None:
                                stop("error")
                            link.close()
                            link = None
                        else:
                            if rollout is not None:
                                answered(obs, reply)
                    if rollout is not None and max_s and now - rollout["start_ns"] > max_s * 1e9:
                        stop(f"ROLLOUT_MAX_S {max_s:g} s")
                    if rollout is not None and future is None and now >= next_obs:
                        if (obs := observe(now)) is not None:
                            sent_obs = obs
                            future = pool.submit(link.infer, rollout["steps"] + 1, vpp_cameras, obs["images"],
                                                 obs["state"], instruction)
                    beat(now)
                    while pending and pending[0][2] <= now:
                        close(*pending.pop(0)[:2])
                    if now - last_status >= 1e9:
                        last_status = now
                        stale = [c.role for c in cams if now - c.last_ns >= 5e8]
                        warn = f"  ⚠ {'/'.join(stale)} camera not delivering frames, robot locked" if stale else ""
                        if rollout is not None:
                            s = (now - rollout["start_ns"]) / 1e9
                            lat = f", latency {np.median(latencies) / 1e6:.0f} ms" if latencies else ""
                            console.status(f"● Rollout {rollout['episode'].name}  {int(s // 60):02d}:{int(s % 60):02d}"
                                           f"  {rollout['steps']} policy steps{lat}{warn}")
                        else:
                            console.status((f"  Waiting: {busy()}" if busy() else
                                            "Ready: press Space to start a rollout") + warn)
        except BaseException:
            end_reason = "error"
            raise
        finally:
            if rollout is not None:
                stop(end_reason)
            for episode, reason, _ in pending:
                save(episode, reason)
            pool.shutdown(wait=False, cancel_futures=True)
            for c in cams:
                c.stop()
            if link is not None:
                link.close()
            meta["end_host_mono_ns"] = time.monotonic_ns()
            save_meta()
            console.say(f"Exited. {len(meta['rollouts'])} rollout(s) this run in {run_dir}")


def check():
    """--check: is the server up, and does it serve a model trained on what deploy.yml prepares?"""
    import vpp
    env = node_env(HERE / "deploy.yml", "policy")
    addr = os.getenv("POLICY_ADDR", "127.0.0.1:47110")
    try:
        link = Link(addr, 5)
    except (OSError, ConnectionError) as exc:
        print(f"  ✗ no policy server at {addr} ({exc})")
        return 2
    link.close()
    m = link.model
    cameras = [c for c in env.get("VPP_CAMERAS", "").split(",") if c]
    crops = vpp.parse_crops(env.get("VPP_CROP", ""))
    problems = mismatches(m, cameras, crops, int(env.get("VPP_SIZE", "256")), int(env.get("VPP_FPS", "10")))
    print(f"  {'✗' if problems else '✓'} policy server at {addr}: {m['checkpoint']} [{m['weights']}] epoch {m['epoch']}"
          f" on {m['device']}; cameras {m['cameras']}, crop {m['crop']}, {m['size']} px, {m['fps']} fps")
    for p in problems:
        print(f"    differs from deploy.yml: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        raise SystemExit(check())
    main()
