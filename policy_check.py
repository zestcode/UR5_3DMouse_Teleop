"""Replay a recorded episode through the policy server as the live policy node would send it, and compare the replies
with what the operator did. The whole path except the robot and the cameras: policy.prepare on the raw camera
frames (crop before the resize, BGR -> RGB), policy.robot_state from the run's robot.jsonl, the wire (policy.Link),
the server. Checked against the episode's VPP export: the prepared images against its video frames (they differ
only by the H.264 compression if crop, resize and colour order are right), the state against its states, and the
predicted chunk against its actions (position / rotation error, gripper agreement). Use a val episode (episode
numbers divisible by 10): the model did not train on it.

  python policy_check.py OUT_DIR/<task>/<NNN_episode> [--steps 30]
The server at POLICY_ADDR (default 127.0.0.1:47110): on this machine (bash ../vpp/deploy_ur5_server.sh start), or
on pegasus through deploy.sh's tunnel or
  ssh -N -L 47110:127.0.0.1:47110 pegasus &
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import policy
import vpp
from common import rot6d_to_mat


def read_frames(path, wanted):
    """{index: BGR frame} of the frame indices `wanted` of a video, read in one pass."""
    import cv2
    cap, out, index = cv2.VideoCapture(str(path)), {}, -1
    try:
        for want in sorted(set(wanted)):
            while index < want:
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"{path} has {index + 1} frames, frame {want} needed")
                index += 1
            out[want] = image
    finally:
        cap.release()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Replay a recorded episode through the VPP policy server.")
    p.add_argument("episode", type=Path, help="OUT_DIR/<task>/<NNN_episode>")
    p.add_argument("--steps", type=int, default=30, help="policy steps, evenly spread over the episode")
    p.add_argument("--max-gap-ms", type=float, default=50)
    args = p.parse_args(argv)
    episode = args.episode.resolve()
    task, number = episode.parent, vpp.episode_number(episode.name)
    ann_path = next((task / "_vpp" / "annotation" / s / f"{number}.json" for s in ("train", "val")
                     if (task / "_vpp" / "annotation" / s / f"{number}.json").exists()), None)
    if ann_path is None:
        raise SystemExit(f"{episode.name} has no VPP export in {task / '_vpp'} (python vpp.py {task} {episode.name})")
    ann = json.loads(ann_path.read_text())
    info = json.loads((episode / "episode.json").read_text())
    cameras, crops = ann["cameras"], {c: tuple(b) for c, b in ann["crop"].items()}
    addr = os.getenv("POLICY_ADDR", "127.0.0.1:47110")
    link = policy.Link(addr, 30)
    m = link.model
    size = m["size"]
    if problems := policy.mismatches(m, cameras, crops, size, ann["fps"]):
        raise SystemExit("the export and the served model differ: " + "; ".join(problems))
    A = m["action_seq_len"]
    print(f"{episode.name} ({ann_path.parent.name} {number}, {ann['video_length']} steps, \"{ann['texts'][0]}\") "
          f"-> {addr}: {m['checkpoint']} [{m['weights']}] epoch {m['epoch']}")

    t = np.asarray(ann["source"]["t_ns"], dtype=np.int64)
    steps = np.unique(np.linspace(0, len(t) - A, args.steps).astype(int))
    robot = vpp.load_robot(task / "_runs" / info["run"] / "robot.jsonl", int(t[0]) - 1e9, int(t[-1]) + 1e9)
    raw = {c: read_frames(episode / f"{c}.mp4", [ann["source"]["frame_index"][c][s] for s in steps]) for c in cameras}
    exported = {c: read_frames(task / "_vpp" / ann["videos"][k]["video_path"], steps) for k, c in enumerate(cameras)}
    states, actions = np.asarray(ann["states"]), np.asarray(ann["actions"])
    rows = []
    for s in steps:
        images = [policy.prepare(raw[c][ann["source"]["frame_index"][c][s]], size, crops.get(c)) for c in cameras]
        state = policy.robot_state(robot["t"], robot["tcp"], robot["grip"][:, 0], int(t[s]), args.max_gap_ms * 1e6)
        reply = link.infer(int(s), cameras, images, state, ann["texts"][0])
        pred, gt = np.asarray(reply["actions"]), actions[s:s + A]
        rel = rot6d_to_mat(pred[:, 3:9]).transpose(0, 2, 1) @ rot6d_to_mat(gt[:, 3:9])
        rows.append({
            "image_diff": [np.abs(i.astype(int) - exported[c][s][:, :, ::-1].astype(int)).mean()
                           for c, i in zip(cameras, images)],
            "state_diff": np.abs(state - states[s]).max(),
            "pos_mm": np.linalg.norm(pred[:, :3] - gt[:, :3], axis=1).mean() * 1e3,
            "rot_deg": np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))).mean(),
            "grip": ((pred[:, 9] > .5) == (gt[:, 9] > .5)).mean(),
            "infer_ms": reply["infer_ms"], "round_trip_ms": (reply["reply_ns"] - reply["sent_ns"]) / 1e6})
    link.close()
    diff = np.mean([r["image_diff"] for r in rows], axis=0)
    print("prepared images vs the export's frames, mean |difference| (0..255): "
          + ", ".join(f"{c} {d:.2f}" for c, d in zip(cameras, diff)) + "  (a few levels: H.264; more: other crop)")
    print(f"state vs the export's states: max |difference| {max(r['state_diff'] for r in rows):.2e}")
    print(f"chunk vs the operator's actions, mean over {len(rows)} steps: position {np.mean([r['pos_mm'] for r in rows]):.1f} mm, "
          f"rotation {np.mean([r['rot_deg'] for r in rows]):.2f} deg, gripper agreement {np.mean([r['grip'] for r in rows]):.3f}"
          f"  (training val: {(m.get('val') or {}).get('pos_mm', float('nan')):.1f} mm, "
          f"gripper {(m.get('val') or {}).get('grip_acc', float('nan')):.3f})")
    print(f"timing: inference {np.median([r['infer_ms'] for r in rows]):.0f} ms, round trip "
          f"{np.median([r['round_trip_ms'] for r in rows]):.0f} ms (median; max {max(r['round_trip_ms'] for r in rows):.0f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
