"""UNIT tests of vpp.py (the VPP export) and vpp_preview.py on their own, with synthetic data (no Dora, no cameras,
no robot): step grid and nearest frames, rotation and gripper conversion, state / action interpolation against
analytic values, crop boxes and frame cropping / padding, one episode exported from synthetic videos (which raw frame lands at each step, the
annotation, H.264 and the mp4v fallback), removal, reading a run's robot.jsonl, the CLI on a synthetic task folder,
and the side-by-side preview."""
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

import recorder as r
import vpp
import vpp_preview

MS = 1_000_000


def rotz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def rot6d(R):  # first two rows of R, as the ur5 node records it
    return R[:2, :].reshape(6)


def pose(x, angle):
    return np.r_[[x, 0., 0.], rot6d(rotz(angle) @ np.diag([1., -1., -1.]))]  # TCP pointing down, turned about z


def robot(n=400, hole_at=None):
    """125 Hz records over n * 8 ms: x = 1 mm per record, rotation 0.01 rad per record about z, target 5 mm ahead
    in x; gripper position k / n, target 1 from record n / 2 on. hole_at: 100 ms without records after that index."""
    t = np.arange(n, dtype=np.int64) * 8 * MS
    if hole_at is not None:
        t[hole_at + 1:] += 100 * MS
    k = np.arange(n)
    return vpp.robot_arrays(t, [pose(i * .001, i * .01) for i in k], [pose(i * .001 + .005, i * .01) for i in k],
                            np.c_[k / n, (k >= n // 2).astype(float)])


def write_video(path, n, size=(640, 480)):
    """n frames; frame k shows k in binary: six vertical stripes, bright = 1 (survives compression and resizing)."""
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    for k in range(n):
        image = np.full((size[1], size[0], 3), 25, np.uint8)
        for b in range(6):
            if k >> b & 1:
                image[:, b * size[0] // 6:(b + 1) * size[0] // 6] = 230
        out.write(image)
    out.release()


def frame_ids(path, stripes=6):
    """The frame number shown by each frame of a write_video() video (also after vpp.square); stripes < 6: of a crop
    that kept only the first `stripes` stripes, the number modulo 2 ** stripes."""
    cap = cv2.VideoCapture(str(path))
    ids = []
    while True:
        ok, image = cap.read()
        if not ok:
            return ids
        w = image.shape[1]
        ids.append(sum(1 << b for b in range(stripes) if image[:, (2 * b + 1) * w // (2 * stripes)].mean() > 128))


def make_episode(root, name="010_episode"):
    """fixed: 30 fps from 0 ms; wrist: every 36 ms from 13 ms, so its frames fall elsewhere."""
    episode = root / name
    episode.mkdir()
    times = {"fixed": (np.arange(45) * 1e9 / 30).astype(np.int64), "wrist": 13 * MS + np.arange(41) * 36 * MS}
    for c, t in times.items():
        write_video(episode / f"{c}.mp4", len(t))
    return episode, times


class TestGridAndFrames(unittest.TestCase):
    def test_grid_covers_the_common_span(self):
        a = np.arange(0, 1000 * MS, 33 * MS)  # 0 .. 990 ms
        b = np.arange(20 * MS, 1200 * MS, 33 * MS)  # 20 .. 1175 ms
        t = vpp.grid([a, b], 100 * MS)
        np.testing.assert_array_equal(t, np.arange(20, 991, 100) * MS)  # from the later start to the earlier end
        self.assertEqual(len(vpp.grid([a, b + 2000 * MS], 100 * MS)), 0)  # never together

    def test_nearest_frame(self):
        times = np.array([0, 10, 20, 40]) * MS
        t = np.array([-5, 4, 6, 15, 29, 31, 100]) * MS
        np.testing.assert_array_equal(vpp.nearest(times, t), [0, 0, 1, 1, 2, 3, 3])  # a tie (15) takes the earlier
        np.testing.assert_array_equal(vpp.nearest(times[:1], t), [0] * len(t))


class TestConversions(unittest.TestCase):
    def test_quaternion_to_rot6d_matches_the_matrix_rows(self):
        rng = np.random.default_rng(1)
        for _ in range(100):
            R = r.rot6d_to_mat(rng.normal(size=6))
            np.testing.assert_allclose(vpp.quat_to_rot6d(r.mat_to_quat(R)), rot6d(R), atol=1e-12)

    def test_gripper_values(self):
        g = lambda **kw: {"gripper": {"mode": "stopped", "pos": 51, "obj": 3, "t_ns": 1, **kw}}
        self.assertEqual(vpp.gripper_values({"gripper": None}), (0., 0.))
        self.assertEqual(vpp.gripper_values(g(pos=None, target=None)), (0., 0.))  # not polled yet
        # recorded target: commanded position; None = stopped with the button, holds
        self.assertEqual(vpp.gripper_values(g(mode="closing", obj=0, target=255)), (.2, 1.))
        self.assertEqual(vpp.gripper_values(g(obj=2, target=255)), (.2, 1.))  # on an object: still squeezing
        self.assertEqual(vpp.gripper_values(g(target=None)), (.2, .2))
        self.assertEqual(vpp.gripper_values(g(mode="opening", obj=0, target=0)), (.2, 0.))
        # older records without target: from mode / obj
        self.assertEqual(vpp.gripper_values(g(mode="closing", obj=0)), (.2, 1.))
        self.assertEqual(vpp.gripper_values(g(mode="opening", obj=0)), (.2, 0.))
        self.assertEqual(vpp.gripper_values(g(obj=2)), (.2, 1.))
        self.assertEqual(vpp.gripper_values(g(obj=1)), (.2, 0.))
        self.assertEqual(vpp.gripper_values(g(obj=3)), (.2, .2))

    def test_states_and_actions_against_analytic_values(self):
        rob = robot()
        t = np.array([4, 100, 1604]) * MS  # records every 8 ms: midway, 1/2 past 12, 1/2 past 200
        keep, states, actions = vpp.trajectory(rob, t, 50 * MS)
        self.assertEqual((keep, states.shape, actions.shape), (slice(0, 3), (3, 10), (3, 10)))
        k = t / (8 * MS)  # fractional record index
        np.testing.assert_allclose(states[:, 0], k * .001, atol=1e-12)
        np.testing.assert_allclose(actions[:, 0], k * .001 + .005, atol=1e-12)
        for i in range(3):
            np.testing.assert_allclose(states[i, 3:9], pose(0, k[i] * .01)[3:], atol=1e-12)  # slerp about z
            np.testing.assert_allclose(actions[i, 3:9], states[i, 3:9], atol=1e-12)
        np.testing.assert_allclose(states[:, 9], np.floor(k) / 400)  # gripper: the last record at the step
        np.testing.assert_array_equal(actions[:, 9], [0, 0, 1])

    def test_steps_without_robot_pose(self):
        keep, states, _ = vpp.trajectory(robot(), np.array([-5, 10, 20, 4000, 5000]) * MS, 50 * MS)
        self.assertEqual((keep, len(states)), (slice(1, 3), 2))  # before the first / after the last record: dropped
        np.testing.assert_allclose(states[:, 0], [.00125, .0025], atol=1e-12)
        with self.assertRaisesRegex(ValueError, r"no robot pose at 1 of 3 steps \(first 0.44 s in\)"):
            vpp.trajectory(robot(hole_at=50), np.array([10, 450, 900]) * MS, 50 * MS)  # 400 .. 508 ms: no record
        with self.assertRaisesRegex(ValueError, "no robot pose at any of the 2 steps"):
            vpp.trajectory(robot(), np.array([4000, 5000]) * MS, 50 * MS)

    def test_square_pads_wide_frames_to_4_3(self):
        wide = np.full((360, 640, 3), 200, np.uint8)  # 16:9
        out = vpp.square(wide, 256)
        self.assertEqual(out.shape, (256, 256, 3))
        self.assertEqual(out[:20].max(), 0)  # 60 of 480 rows of black on top -> 32 of 256
        self.assertEqual(out[40:216].min(), 200)
        self.assertEqual(vpp.square(np.full((480, 640, 3), 7, np.uint8), 256).min(), 7)  # 4:3: no bars

    def test_parse_crops(self):
        self.assertEqual(vpp.parse_crops(""), {})
        self.assertEqual(vpp.parse_crops(vpp.CROP), {"fixed": (150, 100, 500, 450)})
        self.assertEqual(vpp.parse_crops(" fixed:150,100,500,450  wrist:0,0,64,48 "),
                         {"fixed": (150, 100, 500, 450), "wrist": (0, 0, 64, 48)})
        for bad in ("fixed", "fixed:1,2,3", "fixed:a,0,1,1", ":0,0,1,1", "fixed:5,0,5,1", "fixed:0,9,1,2",
                    "fixed:-1,0,1,1", "fixed:0,0,1,1 fixed:0,0,2,2"):
            with self.assertRaises(ValueError, msg=bad):
                vpp.parse_crops(bad)

    def test_square_crops_before_resizing(self):
        image = np.zeros((480, 640, 3), np.uint8)
        image[100:450, 150:500] = 200  # the box
        image[100:450, 150] = 90  # its first column
        out = vpp.square(image, 350, (150, 100, 500, 450))  # 350 x 350: cut, no resampling
        self.assertEqual(out.shape, (350, 350, 3))
        self.assertTrue((out[:, 0] == 90).all() and (out[:, 1:] == 200).all())
        self.assertGreaterEqual(vpp.square(image, 256, (150, 100, 500, 450)).min(), 90)  # nothing from outside (0)
        wide = vpp.square(np.full((480, 640, 3), 200, np.uint8), 256, (0, 0, 640, 240))  # 8:3: padded to 4:3
        self.assertEqual((wide[:50].max(), wide[80:176].min()), (0, 200))  # 120 of 480 rows black on top -> 64
        with self.assertRaisesRegex(ValueError, "outside the 640 x 480 frame"):
            vpp.square(image, 256, (0, 0, 641, 480))


class TestExport(unittest.TestCase):
    def test_export_writes_the_vpp_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode, times = make_episode(Path(tmp))
            root = Path(tmp) / "_vpp"
            ann = vpp.export(root, episode, ["fixed", "wrist"], times, robot(), "pick up the cup", 10, 256, 10,
                             50 * MS, source={"task": "t", "run": "r"})
            # 13 .. 1453 ms (fixed ends at 1466.7, wrist at 1453): steps 13, 113, ..., 1413
            t = 13 * MS + np.arange(15) * 100 * MS
            self.assertEqual(ann["source"]["t_ns"], t.tolist())
            self.assertEqual((ann["episode_id"], ann["video_length"], ann["texts"]), (10, 15, ["pick up the cup"]))
            self.assertEqual(ann["videos"], [{"video_path": "videos/val/10/0.mp4"}, {"video_path": "videos/val/10/1.mp4"}])
            self.assertEqual(ann["latent_videos"][1], {"latent_video_path": "latent_videos/val/10/1.pt"})
            self.assertEqual((len(ann["states"]), len(ann["states"][0]), len(ann["actions"])), (15, 10, 15))
            np.testing.assert_allclose(np.array(ann["states"])[:, 0], t / (8 * MS) * .001, atol=1e-6)
            fixed = np.round(t / (1e9 / 30)).astype(int)  # nearest 30 fps frame
            wrist = np.round((t - 13 * MS) / (36 * MS)).astype(int)  # never half way: 25 i / 9
            self.assertEqual(ann["source"]["frame_index"], {"fixed": fixed.tolist(), "wrist": wrist.tolist()})
            self.assertLessEqual(ann["source"]["max_offset_ms"]["fixed"], 1e3 / 60)
            self.assertLessEqual(ann["source"]["max_offset_ms"]["wrist"], 18)
            for k, picked in enumerate((fixed, wrist)):
                self.assertEqual(frame_ids(root / ann["videos"][k]["video_path"]), picked.tolist())  # raw frame
            cap = cv2.VideoCapture(str(root / "videos/val/10/0.mp4"))
            self.assertEqual((cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                              cap.get(cv2.CAP_PROP_FPS)), (256, 256, 10))
            self.assertTrue(vpp.has_libx264(), "this machine's ffmpeg has libx264: the test expects H.264")
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=codec_name,codec_tag_string,pix_fmt", "-of", "csv=p=0",
                                    str(root / "videos/val/10/0.mp4")], capture_output=True, text=True).stdout
            self.assertEqual((ann["video_codec"], probe.strip()), ("h264", "h264,avc1,yuv420p"))  # browsers play it
            data = (root / "videos/val/10/0.mp4").read_bytes()
            self.assertLess(data.find(b"moov"), data.find(b"mdat"))  # index first: plays while loading
            self.assertEqual(json.loads((root / "annotation/val/10.json").read_text()), ann)
            self.assertEqual(sorted(p.name for p in (root / "annotation").iterdir()), ["train", "val"])
            self.assertEqual([p.name for p in (root / "annotation/val").iterdir()], ["10.json"])  # no .tmp left

    def test_split_reexport_and_remove(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode, times = make_episode(Path(tmp), "003_pick")
            root = Path(tmp) / "_vpp"
            args = (["fixed", "wrist"], times, robot(), "x", 10, 256)
            self.assertEqual(vpp.export(root, episode, *args, 10)["videos"][0]["video_path"], "videos/train/3/0.mp4")
            (root / "latent_videos/train/3").mkdir(parents=True)  # as vpp_latents.py would
            vpp.export(root, episode, *args, 3)  # every 3rd: now val; the train export and its latents are replaced
            self.assertFalse((root / "annotation/train/3.json").exists())
            self.assertFalse((root / "latent_videos/train/3").exists())
            self.assertTrue((root / "annotation/val/3.json").exists())
            vpp.remove(root, 3)
            self.assertEqual([p for p in root.rglob("*") if p.is_file()], [])
            vpp.remove(root, 3)  # nothing there: fine

    def test_export_crops_a_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode, times = make_episode(Path(tmp))
            root = Path(tmp) / "_vpp"
            ann = vpp.export(root, episode, ["fixed", "wrist"], times, robot(), "x", crops={"fixed": (0, 40, 320, 400)})
            self.assertEqual(ann["crop"], {"fixed": [0, 40, 320, 400]})
            picked = ann["source"]["frame_index"]
            cap = cv2.VideoCapture(str(root / ann["videos"][0]["video_path"]))
            self.assertEqual((cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), (256, 256))
            # fixed kept the left half: stripes 0 .. 2 fill the frame; wrist: all six
            self.assertEqual(frame_ids(root / ann["videos"][0]["video_path"], 3), [i % 8 for i in picked["fixed"]])
            self.assertEqual(frame_ids(root / ann["videos"][1]["video_path"]), picked["wrist"])
            with self.assertRaisesRegex(ValueError, r"crop for \['top'\]: not among the cameras"):
                vpp.export(root, episode, ["fixed", "wrist"], times, robot(), "x", crops={"top": (0, 0, 1, 1)})
            with self.assertRaisesRegex(ValueError, "outside the 640 x 480 frame"):
                vpp.export(root, episode, ["fixed", "wrist"], times, robot(), "x", crops={"wrist": (0, 0, 640, 481)})

    def test_cameras_that_never_overlap_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode, times = make_episode(Path(tmp))
            times["wrist"] = times["wrist"] + 5000 * MS
            with self.assertRaisesRegex(ValueError, "less than two steps"):
                vpp.export(Path(tmp) / "_vpp", episode, ["fixed", "wrist"], times, robot(), "x")


    def test_mp4v_without_libx264(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode, times = make_episode(Path(tmp))
            saved, vpp._LIBX264 = vpp._LIBX264, False
            try:
                ann = vpp.export(Path(tmp) / "_vpp", episode, ["fixed", "wrist"], times, robot(), "x")
            finally:
                vpp._LIBX264 = saved
            self.assertEqual(ann["video_codec"], "mp4v")
            self.assertEqual(frame_ids(Path(tmp) / "_vpp" / ann["videos"][1]["video_path"]),
                             ann["source"]["frame_index"]["wrist"])

    def test_preview_puts_the_cameras_side_by_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode, times = make_episode(Path(tmp))
            root = Path(tmp) / "_vpp"
            ann = vpp.export(root, episode, ["fixed", "wrist"], times, robot(), "pick up the cup")
            self.assertEqual(vpp_preview.preview(root, ann, Path(tmp) / "p.mp4"), 15)
            cap = cv2.VideoCapture(str(Path(tmp) / "p.mp4"))
            self.assertEqual((cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                              cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), (15, 1024, 512 + vpp_preview.BAND))
            for k in range(15):
                ok, sheet = cap.read()
                halves = [sheet[vpp_preview.BAND:, :512], sheet[vpp_preview.BAND:, 512:]]
                ids = [sum(1 << b for b in range(6) if h[:, (2 * b + 1) * 512 // 12].mean() > 128) for h in halves]
                self.assertEqual(ids, [ann["source"]["frame_index"][c][k] for c in ("fixed", "wrist")])


class TestRunFiles(unittest.TestCase):
    def write_run(self, path, n=400, close_from=50):
        """robot() records as robot.jsonl lines; gripper as recorded before `target`: opening, closing from 400 ms."""
        rob = robot(n)
        with path.open("w") as f:
            for k in range(n):
                grip = {"mode": "closing" if k >= close_from else "opening", "pos": round(255 * k / n), "obj": 0,
                        "t_ns": 0}
                f.write(json.dumps({"t_ns": int(rob["t"][k]), "tcp_pose": rob["tcp"][k].tolist(),
                                    "target_pose": rob["target"][k].tolist(), "gripper": grip}) + "\n")

    def test_load_robot_keeps_the_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "robot.jsonl"
            self.write_run(path)
            rob = vpp.load_robot(path, 80 * MS, 160 * MS)
            np.testing.assert_array_equal(rob["t"], np.arange(10, 21) * 8 * MS)
            np.testing.assert_allclose(rob["tcp"][:, 0], np.arange(10, 21) * .001)
            self.assertEqual(rob["grip"].shape, (11, 2))

    def test_cli_exports_a_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp) / "pick_cup"
            (task / "_runs" / "run1").mkdir(parents=True)
            self.write_run(task / "_runs" / "run1" / "robot.jsonl")
            (task / "task.json").write_text(json.dumps({"instruction": "pick up the cup"}))
            episode, times = make_episode(task, "001_episode")
            for c, t in times.items():
                with (episode / f"{c}_frames.csv").open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["index", "host_receive_mono_ns", "frame_mono_ns"])
                    w.writerows((i, v + 15 * MS, v) for i, v in enumerate(t.tolist()))
            info = {"task": "pick_cup", "run": "run1", "completed": True, "end_reason": "stopped"}
            (episode / "episode.json").write_text(json.dumps(info))
            (task / "002_episode").mkdir()  # the run ended while recording it
            (task / "002_episode" / "episode.json").write_text(json.dumps({**info, "completed": False}))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(vpp.main([str(task)]), 0)
            self.assertIn("002_episode: skipped (incomplete)", out.getvalue())
            ann = json.loads((task / "_vpp/annotation/train/1.json").read_text())
            self.assertEqual((ann["texts"], ann["video_length"], ann["source"]["run"]), (["pick up the cup"], 15, "run1"))
            self.assertEqual(ann["crop"], {"fixed": [150, 100, 500, 450]})  # the default, vpp.CROP
            self.assertEqual(np.array(ann["actions"])[:, 9].tolist(), [0.] * 4 + [1.] * 11)  # steps from 413 ms
            with contextlib.redirect_stdout(out):
                self.assertEqual(vpp.main([str(task), "001_episode", "--instruction", "lift the cup",
                                           "--out", str(Path(tmp) / "elsewhere")]), 0)
            self.assertEqual(json.loads((Path(tmp) / "elsewhere/annotation/train/1.json").read_text())["texts"],
                             ["lift the cup"])
            with contextlib.redirect_stdout(out):
                self.assertEqual(vpp.main([str(task), "001_episode", "--crop", ""]), 0)
                self.assertEqual(vpp.main([str(task), "001_episode", "--crop", "top:0,0,1,1"]), 1)
            self.assertIn("001_episode: FAILED: crop for ['top']: not cameras of the episode", out.getvalue())
            self.assertEqual(json.loads((task / "_vpp/annotation/train/1.json").read_text())["crop"], {})
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                vpp.main([str(task), "--crop", "fixed:1,2,3"])


if __name__ == "__main__":
    unittest.main()
