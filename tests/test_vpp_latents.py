"""UNIT tests of vpp_latents.py with a stub VAE (no diffusers, no SVD weights, CPU): the frames reach the VAE as RGB
in [-1, 1] in batches, the latents are scaled by scaling_factor and saved where the annotation says, existing ones
are skipped. Needs torch, which the teleop environment does not have: skipped there; run it in VPP's environment
(or any with torch + opencv): python -m unittest tests.test_vpp_latents"""
import json
from pathlib import Path
import tempfile
import types
import unittest

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None


class StubVAE:
    """encode(x) -> latent_dist.sample() = per-frame channel means of x, as (B, 3, 1, 1); records the inputs."""
    device = "cpu"
    config = types.SimpleNamespace(scaling_factor=0.5)

    def __init__(self):
        self.calls = []

    def encode(self, x):
        self.calls.append(x.clone())
        mean = x.mean(dim=(2, 3), keepdim=True)
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(sample=lambda: mean.clone()))


@unittest.skipIf(torch is None, "needs torch (VPP's environment)")
class TestLatents(unittest.TestCase):
    def test_encode_folder(self):
        import vpp_latents
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "videos/train/1").mkdir(parents=True)
            (root / "annotation/train").mkdir(parents=True)
            for k, bgr in enumerate([(0, 0, 255), (255, 0, 0)]):  # camera 0 red, camera 1 blue
                out = cv2.VideoWriter(str(root / f"videos/train/1/{k}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                      (256, 256))
                for _ in range(5):
                    out.write(np.full((256, 256, 3), bgr, np.uint8))
                out.release()
            ann = {"videos": [{"video_path": f"videos/train/1/{k}.mp4"} for k in range(2)], "video_length": 5,
                   "latent_videos": [{"latent_video_path": f"latent_videos/train/1/{k}.pt"} for k in range(2)]}
            (root / "annotation/train/1.json").write_text(json.dumps(ann))
            vae = StubVAE()
            self.assertEqual(vpp_latents.encode_folder(vae, root, batch_size=2), 2)
            self.assertEqual([len(x) for x in vae.calls], [2, 2, 1] * 2)  # 5 frames in batches of 2
            x = torch.cat(vae.calls[:3])
            self.assertEqual(tuple(x.shape), (5, 3, 256, 256))
            self.assertGreaterEqual(float(x.min()), -1)
            self.assertLessEqual(float(x.max()), 1)
            red = torch.load(root / "latent_videos/train/1/0.pt", weights_only=True)
            self.assertEqual(tuple(red.shape), (5, 3, 1, 1))
            np.testing.assert_allclose(red[:, :, 0, 0].numpy(), [[.5, -.5, -.5]] * 5, atol=.03)  # RGB, x 0.5
            blue = torch.load(root / "latent_videos/train/1/1.pt", weights_only=True)
            np.testing.assert_allclose(blue[:, :, 0, 0].numpy(), [[-.5, -.5, .5]] * 5, atol=.03)
            self.assertEqual(vpp_latents.encode_folder(vae, root), 0)  # both there: skipped
            self.assertEqual(vpp_latents.encode_folder(vae, root, force=True), 2)
            self.assertEqual(sorted(p.name for p in (root / "latent_videos/train/1").iterdir()), ["0.pt", "1.pt"])


if __name__ == "__main__":
    unittest.main()
