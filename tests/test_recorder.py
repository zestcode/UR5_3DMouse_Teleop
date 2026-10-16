"""UNIT tests of recorder.py: rotation conversions, pose interpolation, plan parsing, one Episode written from
synthetic frames (no Dora, no cameras, no robot)."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import recorder as r


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def rot6d(R):  # first two rows of R, as the ur5 node records it
    return R[:2, :].reshape(6)


class TestRotations(unittest.TestCase):
    def test_rot6d_and_quaternion_round_trip(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            q2 = r.mat_to_quat(r.rot6d_to_mat(rot6d(quat_to_mat(q))))
            self.assertLess(min(np.abs(q2 - q).max(), np.abs(q2 + q).max()), 1e-9)
            self.assertGreaterEqual(q2[3], 0)


class TestInterpolate(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(100, dtype=np.int64) * 8_000_000  # 125 Hz
        self.p = np.array([np.r_[[k * .001, 0, 0], rot6d(rotz(k * .01))] for k in range(100)])

    def test_midpoints_match_analytic_values(self):
        ok, gap, xyz, q = r.interpolate(self.t, self.p, [4_000_000, 12_000_000], 50e6)
        self.assertTrue(ok.all())
        np.testing.assert_allclose(gap, [8, 8])
        np.testing.assert_allclose(xyz[:, 0], [.0005, .0015], atol=1e-12)
        np.testing.assert_allclose(2 * np.arctan2(q[:, 2], q[:, 3]), [.005, .015], atol=1e-12)

    def test_outside_or_wide_gap_is_invalid(self):
        t = np.r_[self.t[:10], self.t[10:] + 100_000_000]  # 100 ms hole after sample 9
        ok, _, xyz, _ = r.interpolate(t, self.p, [-1, self.t[-1] + 200_000_000, t[9] + 1, 4_000_000], 50e6)
        self.assertEqual(ok.tolist(), [False, False, False, True])
        self.assertTrue(np.isnan(xyz[:3]).all())

    def test_too_few_robot_records(self):
        ok, _, _, _ = r.interpolate(self.t[:1], self.p[:1], [0], 50e6)
        self.assertFalse(ok.any())


class TestPlan(unittest.TestCase):
    def test_labels(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
            f.write("# comment\nR1 a\n\n  pick/place  # note\n")
            f.flush()
            self.assertEqual(r.read_plan(f.name), ["R1_a", "pick_place"])
        self.assertEqual(r.read_plan(""), [])
        self.assertEqual(r.safe_label(" / "), "episode")


class TestEpisode(unittest.TestCase):
    def test_video_frames_and_poses_line_up(self):
        class Cam:
            role = "wrist"
        with tempfile.TemporaryDirectory() as tmp:
            ep = r.Episode(Path(tmp), 1, "x", [Cam()], (64, 48), 30)
            t0 = 1_000_000_000
            for n in range(10):
                ep.add("wrist", np.full((48, 64, 3), n * 20, np.uint8),
                       {"frame_number": n + (n >= 5), "timestamp_ms": 0., "timestamp_domain": "mock",
                        "host_receive_mono_ns": t0 + n * 33_333_333, "frame_mono_ns": t0 + n * 33_333_333})
            robot_t = t0 - 8_000_000 + np.arange(60, dtype=np.int64) * 8_000_000
            pose = np.array([np.r_[[0, 0, k * .001], rot6d(np.eye(3))] for k in range(60)])
            info = ep.close("stopped", robot_t, pose, pose, 50e6)
            cam = info["cameras"]["wrist"]
            self.assertEqual((cam["frames"], cam["frames_with_pose"], cam["frame_number_gaps"]), (10, 10, 1))
            rows = list(csv.DictReader((Path(tmp) / "001_x" / "wrist_frames.csv").open()))
            self.assertEqual([int(x["index"]) for x in rows], list(range(10)))
            # frame n at t0 + n*33.33 ms -> z = (n*33.33 + 8) ms / 8 ms * 1 mm
            np.testing.assert_allclose([float(x["tcp_z"]) for x in rows],
                                       [(n * 33.333333 + 8) / 8 * .001 for n in range(10)], atol=1e-7)
            import cv2
            video = cv2.VideoCapture(str(Path(tmp) / "001_x" / "wrist.mp4"))
            self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
            self.assertTrue(json.loads((Path(tmp) / "001_x" / "episode.json").read_text())["completed"])


if __name__ == "__main__":
    unittest.main()
