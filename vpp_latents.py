"""SVD VAE latents of a VPP export (the latent_videos/ VPP trains on), computed as VPP's step1_prepare_latent.py does:
RGB frames scaled to [-1, 1], vae.encode(x).latent_dist.sample() * vae.config.scaling_factor, one (T, 4, 32, 32)
tensor per 256 x 256 video, torch.save to the annotation's latent_video_path.
Not part of the teleop runtime: run it in VPP's Python environment (torch, diffusers, opencv) on a GPU machine:
  python vpp_latents.py <SVD model folder, e.g. svd-robot> <task>/_vpp [more VPP folders ...] [--force]
Videos that already have their latent are skipped (--force: encode them all again)."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_video(path):
    """-> (T, H, W, 3) uint8 RGB."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"cannot read {path}")
    return np.stack(frames)


def encode(vae, frames, batch_size):
    import torch
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(vae.device) / 255.0 * 2 - 1
    with torch.no_grad():
        return torch.cat([vae.encode(x[i:i + batch_size]).latent_dist.sample().mul_(vae.config.scaling_factor).cpu()
                          for i in range(0, len(x), batch_size)])


def encode_folder(vae, root, force=False, batch_size=16):
    """Every video of the VPP folder root that has no latent yet (all with force) -> its latent; -> count written."""
    import torch
    written = 0
    for ann_file in sorted(root.glob("annotation/*/*.json")):
        ann = json.loads(ann_file.read_text())
        for video, latent in zip(ann["videos"], ann["latent_videos"]):
            dst = root / latent["latent_video_path"]
            if dst.exists() and not force:
                continue
            frames = read_video(root / video["video_path"])
            if len(frames) != ann["video_length"]:
                raise RuntimeError(f"{video['video_path']}: {len(frames)} frames, video_length {ann['video_length']}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".tmp")
            torch.save(encode(vae, frames, batch_size), tmp)
            tmp.replace(dst)
            written += 1
            print(f"{dst.relative_to(root)}: {len(frames)} frames", flush=True)
    return written


def main():
    p = argparse.ArgumentParser(description="SVD VAE latents for VPP export folders.")
    p.add_argument("model", help="SVD model folder with a vae/ subfolder (e.g. svd-robot)")
    p.add_argument("folders", nargs="+", type=Path, help="VPP folders (<task>/_vpp)")
    p.add_argument("--force", action="store_true", help="encode videos that already have a latent too")
    p.add_argument("--batch-size", type=int, default=16, help="frames per VAE call (VPP uses 64 on 80 GB GPUs)")
    args = p.parse_args()
    from diffusers.models import AutoencoderKLTemporalDecoder
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.model, subfolder="vae").to("cuda")
    for root in args.folders:
        print(f"{root}: {encode_folder(vae, root, args.force, args.batch_size)} latents written")


if __name__ == "__main__":
    main()
