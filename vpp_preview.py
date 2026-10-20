"""Side-by-side preview of VPP exports, to check by eye that the cameras and the robot data line up: per step the
frame of every camera next to each other (2x), and above them the step and its time, each camera's raw frame number
(the row of its <role>_frames.csv), and the gripper position / target. H.264 when ffmpeg has libx264 (plays in
browsers and VS Code), at the export's rate (10 fps = real time).
  python vpp_preview.py <task>/_vpp [episode id ...]      -> <task>/_vpp/preview/<id>.mp4 (default: every episode)
VPP does not read preview/."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from vpp import VideoWriter

BAND = 56  # text rows above the images
SCALE = 2


def text(image, line, y, x=6, color=(255, 255, 255)):
    cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def preview(root, ann, dst):
    """The preview of one annotation of the VPP folder root -> dst; -> number of steps written."""
    caps = [cv2.VideoCapture(str(root / v["video_path"])) for v in ann["videos"]]
    cameras = ann.get("cameras") or [str(k) for k in range(len(caps))]
    source = ann.get("source", {})
    t = np.asarray(source.get("t_ns", []), dtype=np.int64)
    states, actions = np.asarray(ann["states"]), np.asarray(ann["actions"])
    out, n = None, 0
    try:
        for k in range(ann["video_length"]):
            frames = []
            for cap in caps:
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"{ann['videos'][len(frames)]['video_path']}: no frame {k}")
                frames.append(cv2.resize(image, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST))
            w = frames[0].shape[1]
            sheet = np.zeros((BAND + frames[0].shape[0], w * len(frames), 3), np.uint8)
            sheet[BAND:] = np.hstack(frames)
            seconds = (t[k] - t[0]) / 1e9 if len(t) > k else k / ann.get("fps", 10)
            text(sheet, f"{ann['texts'][0]}   episode {ann['episode_id']}   step {k}/{ann['video_length'] - 1}"
                        f"   t = {seconds:6.1f} s", 18)
            for i, role in enumerate(cameras):
                raw = source.get("frame_index", {}).get(role)
                line = role
                if raw is not None and len(t) > k:
                    line += f"  raw frame {raw[k]}"
                text(sheet, line, 40, x=i * w + 6, color=(0, 255, 255))
            gripper = f"gripper pos {states[k, -1]:.2f} target {actions[k, -1]:.2f}"
            text(sheet, gripper, 40, x=sheet.shape[1] - 9 * len(gripper), color=(0, 255, 0))
            if out is None:
                out = VideoWriter(dst, sheet.shape[1::-1], ann.get("fps", 10))
            out.write(sheet)
            n += 1
    finally:
        for cap in caps:
            cap.release()
        if out is not None:
            out.close()
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description="Side-by-side previews of VPP exports.")
    p.add_argument("root", type=Path, help="VPP folder, <task>/_vpp")
    p.add_argument("ids", nargs="*", type=int, help="episode ids (default: all)")
    args = p.parse_args(argv)
    for ann_file in sorted(args.root.glob("annotation/*/*.json"), key=lambda f: int(f.stem)):
        if args.ids and int(ann_file.stem) not in args.ids:
            continue
        ann = json.loads(ann_file.read_text())
        dst = args.root / "preview" / f"{ann['episode_id']}.mp4"
        dst.parent.mkdir(exist_ok=True)
        print(f"{dst}: {preview(args.root, ann, dst)} steps")


if __name__ == "__main__":
    main()
