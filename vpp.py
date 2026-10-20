"""VPP export: a recorded episode in the dataset layout of Video Prediction Policy
(github.com/roboterax/video-prediction-policy), as its step1_prepare_latent.py writes the data of a real robot.

<task>/_vpp/                                one VPP dataset folder (VPP: dataset_dir = OUT_DIR, dataset = <task>/_vpp)
  annotation/<split>/<id>.json              texts, videos, latent_videos, episode_id, video_length, states, actions
  videos/<split>/<id>/<k>.mp4               camera k of VPP_CAMERAS (0 = rgb_static, 1 = rgb_gripper), SIZE x SIZE,
                                            H.264 (VideoWriter: plays in browsers / VS Code; decord reads it)
  latent_videos/<split>/<id>/<k>.pt         SVD VAE latents: not written here (vpp_latents.py, on a GPU machine)
<id> = episode number (007_episode -> 7); <split> = val for every VAL_EVERY-th episode number, train otherwise.
Steps: a uniform FPS grid (VPP: 0.1 s) over the time every camera recorded (and the robot: steps at the start or
end without robot records are left out); per camera the frame nearest to each step; the robot interpolated there.
Frames: cropped to the camera's CROP box if it has one, padded to 4:3 if wider, resized to SIZE x SIZE (as VPP).
state  (10) = TCP xyz (base frame, m) + TCP rot6d (first two rows of R) + gripper position (0 open .. 1 closed)
action (10) = servo target xyz + rot6d (ur5 target_pose: where the SpaceMouse sent the robot) + gripper target
VPP learns action - state (its x-bot dataset: actions[t .. t+n] - states[t]), so both are in one space. rot6d, not
a quaternion: the TCP points down, qw is near 0, and a sign-canonical quaternion jumps between q and -q.
The annotation also has keys VPP does not read: fps, cameras, crop, state_layout, video_codec, source (episode, run,
step times, the raw frame index of each step per camera, the largest frame-to-step offset).

CLI, to (re-)export recorded episodes from their videos, CSVs and the run's robot.jsonl (e.g. other settings; frame
times of older recordings are retimed here as the recorder now does, recorder.retime):
  python vpp.py <task dir> [episode dir name ...] [--cameras fixed,wrist] [--crop "fixed:150,100,500,450"] [--fps 10]
                [--size 256] [--val-every 10]
"""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from recorder import interpolate, retime
from task import default_instruction, read_instruction

STATE_LAYOUT = ["x", "y", "z", "r11", "r12", "r13", "r21", "r22", "r23", "gripper"]
CROP = "fixed:150,100,500,450"  # the CLI default, as VPP_CROP in dataflow.yml: the table and the arm in the fixed view


def parse_crops(text):
    """'role:x0,y0,x1,y1 ...' (space-separated; '' = none) -> {role: (x0, y0, x1, y1)}: frame[y0:y1, x0:x1]
    (pixels, x1 and y1 exclusive), as crop_region.ipynb shows them."""
    crops = {}
    for item in text.split():
        role, _, box = item.partition(":")
        try:
            x0, y0, x1, y1 = map(int, box.split(","))
        except ValueError:
            raise ValueError(f"crop {item!r}: not role:x0,y0,x1,y1") from None
        if not (role and 0 <= x0 < x1 and 0 <= y0 < y1) or role in crops:
            raise ValueError(f"crop {item!r}: needs a new role and 0 <= x0 < x1, 0 <= y0 < y1")
        crops[role] = (x0, y0, x1, y1)
    return crops


def episode_number(name):
    return int(name.split("_", 1)[0])


def split_of(number, val_every):
    return "val" if val_every and number % val_every == 0 else "train"


def grid(frame_times, period_ns):
    """-> step times (int64 ns) every period_ns over the span all cameras recorded (first step = latest first frame)."""
    t0, t1 = max(int(t[0]) for t in frame_times), min(int(t[-1]) for t in frame_times)
    if t1 < t0:
        return np.zeros(0, np.int64)
    return t0 + np.arange((t1 - t0) // period_ns + 1, dtype=np.int64) * period_ns


def nearest(times, t):
    """-> index of the frame nearest to each step time t (times sorted)."""
    times = np.asarray(times, dtype=np.int64)
    j = np.searchsorted(times, t).clip(0, len(times) - 1)
    i = (j - 1).clip(0)
    return np.where(np.abs(times[i] - t) <= np.abs(times[j] - t), i, j)


def quat_to_rot6d(q):
    """(..., 4) quaternion x y z w -> (..., 6) first two rows of R."""
    x, y, z, w = np.moveaxis(np.asarray(q, dtype=float), -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1)


def gripper_values(record):
    """-> (position, target) of an ur5/state record's gripper, 0 = open .. 1 = closed ((0, 0) without a gripper).
    target = the commanded position: closing 1, opening 0, and still 1 once the firmware stopped on an object (it
    keeps squeezing); stopped with the button: where it stopped. Records without `target` (recorded before the
    gripper reported it): from mode, and obj when stopped (2 = on an object while closing, 1 = while opening)."""
    g = record.get("gripper")
    if not g or g.get("pos") is None:
        return 0., 0.
    pos = g["pos"] / 255
    if "target" in g:
        return pos, pos if g["target"] is None else g["target"] / 255
    target = {"closing": 1., "opening": 0.}.get(g.get("mode"))
    return pos, {2: 1., 1: 0.}.get(g.get("obj"), pos) if target is None else target


def robot_arrays(t_ns, tcp, target, grip):
    """The robot as export() takes it: record times, TCP and target poses (xyz + rot6d), gripper (position, target)."""
    return {"t": np.asarray(t_ns, dtype=np.int64), "tcp": np.asarray(tcp, dtype=float).reshape(-1, 9),
            "target": np.asarray(target, dtype=float).reshape(-1, 9), "grip": np.asarray(grip, dtype=float).reshape(-1, 2)}


def trajectory(robot, t, max_gap_ns):
    """-> (keep, states (N, 10), actions (N, 10)) at the step times t[keep] (see the module doc). A robot pose needs
    two bracketing records at most max_gap_ns apart; steps without one at the start or end are dropped (keep), e.g.
    the last frames when the run is stopped while recording; ValueError for one in between, or no pose at all."""
    ok, _, xyz, q = interpolate(robot["t"], robot["tcp"], t, max_gap_ns)
    ok_target, _, target_xyz, target_q = interpolate(robot["t"], robot["target"], t, max_gap_ns)
    ok &= ok_target
    good = np.flatnonzero(ok)
    if not len(good):
        raise ValueError(f"no robot pose at any of the {len(t)} steps")
    keep = slice(good[0], good[-1] + 1)
    if not ok[keep].all():
        bad = np.flatnonzero(~ok[keep]) + good[0]
        raise ValueError(f"no robot pose at {len(bad)} of {len(t)} steps (first {(t[bad[0]] - t[0]) / 1e9:.2f} s in)")
    grip = robot["grip"][(np.searchsorted(robot["t"], t, side="right") - 1).clip(0)]  # last record at the step
    return (keep, np.c_[xyz, quat_to_rot6d(q), grip[:, 0]][keep],
            np.c_[target_xyz, quat_to_rot6d(target_q), grip[:, 1]][keep])


def square(image, size, crop=None):
    """A camera image as VPP sees it: cut to crop (x0, y0, x1, y1) if given, padded to 4:3 if wider (black bars above
    and below, as step1_prepare_latent.py), then resized to size x size. Use the same on the live images when the
    policy runs (the crop: the annotation's)."""
    import cv2
    if crop is not None:
        x0, y0, x1, y1 = crop
        if x1 > image.shape[1] or y1 > image.shape[0]:
            raise ValueError(f"crop {crop} outside the {image.shape[1]} x {image.shape[0]} frame")
        image = image[y0:y1, x0:x1]
    h, w = image.shape[:2]
    pad = round(w * 3 / 4) - h
    if pad > 0:
        image = cv2.copyMakeBorder(image, pad // 2, pad - pad // 2, 0, 0, cv2.BORDER_CONSTANT)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


_LIBX264 = None


def has_libx264():
    global _LIBX264
    if _LIBX264 is None:
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10)
            _LIBX264 = " libx264 " in out.stdout
        except (OSError, subprocess.SubprocessError):
            _LIBX264 = False
    return _LIBX264


class VideoWriter:
    """BGR frames -> an .mp4 file: H.264 (libx264, yuv420p, moov first: plays in browsers and VS Code) through the
    ffmpeg command when it has libx264, else OpenCV's MPEG-4 Part 2 (mp4v: decord / OpenCV read it, browsers and
    VS Code do not). `codec`: "h264" or "mp4v"."""

    def __init__(self, path, size, fps, h264=None):
        self.codec = "h264" if (has_libx264() if h264 is None else h264) else "mp4v"
        self.path, self.size = Path(path), tuple(size)
        if self.codec == "h264":
            self.proc = subprocess.Popen(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{size[0]}x{size[1]}", "-framerate", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        else:
            import cv2
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size)
            if not self.cv.isOpened():
                raise RuntimeError(f"cannot open {path}")

    def write(self, image):
        if image.shape[1::-1] != self.size:
            raise ValueError(f"frame {image.shape[1::-1]} for a {self.size} video")
        if self.codec == "h264":
            self.proc.stdin.write(np.ascontiguousarray(image, dtype=np.uint8).tobytes())
        else:
            self.cv.write(image)

    def close(self):
        if self.codec == "mp4v":
            self.cv.release()
            return
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        error = self.proc.stderr.read().decode(errors="replace").strip()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg could not write {self.path.name}: {error}")


def write_video(src, dst, frame_ids, size, fps, crop=None):
    """Frames frame_ids (ascending, repeats allowed) of the video src -> video dst, each square()d; -> codec."""
    import cv2
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    out = VideoWriter(dst, (size, size), fps)
    try:
        index, image = -1, None
        for want in frame_ids:
            while index < want:
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"{src.name} has {index + 1} frames, frame {want} needed")
                index += 1
            out.write(square(image, size, crop))
    finally:
        cap.release()
        out.close()
    return out.codec


def remove(root, number):
    """Delete the export of episode `number` from the VPP folder root (any split; none there is fine)."""
    for split in ("train", "val"):
        (root / "annotation" / split / f"{number}.json").unlink(missing_ok=True)  # first: VPP finds episodes by it
        for kind in ("videos", "latent_videos"):
            shutil.rmtree(root / kind / split / str(number), ignore_errors=True)


def export(root, episode, cameras, frame_times, robot, instruction, fps=10, size=256, val_every=10,
           max_gap_ns=50e6, source=None, crops=None):
    """Write the recorded episode directory `episode` into the VPP folder root; -> its annotation.
    cameras: roles in VPP camera order; frame_times[role]: frame_mono_ns of each frame of <role>.mp4 (its CSV rows);
    robot: robot_arrays(); crops: parse_crops() (cameras without one: the whole frame). The annotation is written
    last (atomically): an export cut short stays invisible to VPP."""
    crops = crops or {}
    if unknown := sorted(set(crops) - set(cameras)):
        raise ValueError(f"crop for {unknown}: not among the cameras {list(cameras)}")
    number = episode_number(episode.name)
    split = split_of(number, val_every)
    t = grid([frame_times[c] for c in cameras], round(1e9 / fps))
    if len(t) < 2:
        raise ValueError("the cameras recorded together for less than two steps")
    keep, states, actions = trajectory(robot, t, max_gap_ns)
    t = t[keep]
    if len(t) < 2:
        raise ValueError("the robot was recorded for less than two steps")
    picks = {c: nearest(frame_times[c], t) for c in cameras}
    remove(root, number)  # an earlier export of this number (a deleted episode, or a re-export)
    for s in ("train", "val"):  # VPP lists both annotation folders
        (root / "annotation" / s).mkdir(parents=True, exist_ok=True)
    video_dir = root / "videos" / split / str(number)
    video_dir.mkdir(parents=True)
    codecs = {write_video(episode / f"{c}.mp4", video_dir / f"{k}.mp4", picks[c], size, fps, crops.get(c))
              for k, c in enumerate(cameras)}
    ann = {"task": "robot_trajectory_prediction", "texts": [instruction],
           "videos": [{"video_path": f"videos/{split}/{number}/{k}.mp4"} for k in range(len(cameras))],
           "episode_id": number, "video_length": len(t),
           "latent_videos": [{"latent_video_path": f"latent_videos/{split}/{number}/{k}.pt"} for k in range(len(cameras))],
           "states": states.round(7).tolist(), "actions": actions.round(7).tolist(),
           "fps": fps, "cameras": list(cameras), "crop": {c: list(crops[c]) for c in cameras if c in crops},
           "state_layout": STATE_LAYOUT, "video_codec": "/".join(sorted(codecs)),
           "source": {"episode": episode.name, **(source or {}), "t_ns": t.tolist(),
                      "frame_index": {c: picks[c].tolist() for c in cameras},
                      "max_offset_ms": {c: round(float(np.abs(np.asarray(frame_times[c])[picks[c]] - t).max()) / 1e6, 3)
                                        for c in cameras}}}
    path = root / "annotation" / split / f"{number}.json"
    tmp = path.with_name(path.name + ".tmp")  # not *.json: VPP skips it
    tmp.write_text(json.dumps(ann, allow_nan=False))
    tmp.replace(path)
    return ann


def frame_times(csv_path):
    """frame_mono_ns of a <role>_frames.csv, retimed as the recorder does now (older recordings: not yet)."""
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return retime([int(r["frame_mono_ns"]) for r in rows], [int(r["host_receive_mono_ns"]) for r in rows])[0]


def load_robot(path, t_from, t_to):
    """The ur5/state records of a run's robot.jsonl from t_from to t_to (ns) -> robot_arrays()."""
    stamp = re.compile(r'\{"t_ns": (\d+)')  # the first key (ur5.py): skip the other records without parsing them
    records = []
    with path.open() as f:
        for line in f:
            m = stamp.match(line)
            if m is None or t_from <= int(m.group(1)) <= t_to:
                r = json.loads(line)
                if t_from <= r["t_ns"] <= t_to:
                    records.append(r)
    return robot_arrays([r["t_ns"] for r in records], [r["tcp_pose"] for r in records],
                        [r["target_pose"] for r in records], [gripper_values(r) for r in records])


def main(argv=None):
    p = argparse.ArgumentParser(description="Export recorded episodes of a task to the VPP dataset layout.")
    p.add_argument("task", type=Path, help="task folder, OUT_DIR/<task>")
    p.add_argument("episodes", nargs="*", help="episode folder names (default: every completed episode)")
    p.add_argument("--out", type=Path, help="VPP folder (default: <task>/_vpp)")
    p.add_argument("--cameras", default="fixed,wrist", help="camera roles in VPP order (0 = rgb_static)")
    p.add_argument("--crop", default=CROP, help="'role:x0,y0,x1,y1 ...': frame[y0:y1, x0:x1] before the resize "
                                                f"('' = whole frames; default {CROP})")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--max-gap-ms", type=float, default=50)
    p.add_argument("--instruction", help="default: the episode's, else the task's task.json, else the task name")
    args = p.parse_args(argv)
    task, cameras = args.task.resolve(), [c for c in args.cameras.split(",") if c]
    try:
        crops = parse_crops(args.crop)
    except ValueError as exc:
        p.error(str(exc))
    root = args.out or task / "_vpp"
    names = args.episodes or sorted(e.name for e in task.iterdir() if e.is_dir() and re.match(r"\d+_", e.name))
    failed = 0
    for name in names:
        episode = task / name
        try:
            info = json.loads((episode / "episode.json").read_text())
            if not info.get("completed") or info.get("end_reason") == "error":
                print(f"{name}: skipped ({'ended by an error' if info.get('completed') else 'incomplete'})")
                continue
            if unknown := sorted(set(crops) - set(info.get("cameras", cameras))):
                raise ValueError(f"crop for {unknown}: not cameras of the episode")
            times = {c: frame_times(episode / f"{c}_frames.csv") for c in cameras}
            robot = load_robot(task / "_runs" / info["run"] / "robot.jsonl",
                               min(int(v[0]) for v in times.values()) - 1e9, max(int(v[-1]) for v in times.values()) + 1e9)
            instruction = (args.instruction or info.get("instruction") or read_instruction(task)
                           or default_instruction(task.name))
            ann = export(root, episode, cameras, times, robot, instruction, args.fps, args.size, args.val_every,
                         args.max_gap_ms * 1e6, source={"task": info["task"], "run": info["run"]},
                         crops={c: crops[c] for c in cameras if c in crops})
        except (OSError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            print(f"{name}: FAILED: {exc}")
            failed += 1
            continue
        print(f"{name} -> {ann['videos'][0]['video_path'].split('/')[1]} {ann['episode_id']}: "
              f"{ann['video_length']} steps, \"{instruction}\", max frame offset {ann['source']['max_offset_ms']} ms")
    print(f"VPP folder: {root}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
