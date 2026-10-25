"""UNIT tests of policy.py on its own (no Dora, no cameras, no robot, no model): the wire format and the client
against the fake server, the observation (crop before the resize, BGR -> RGB), the state at a frame time against
analytic values, the training-data check, the replanning time, the camera sink, and that deploy.yml has the
robot and camera settings of dataflow.yml."""
import socket
import unittest

import numpy as np

import policy
import vpp
from tests.fake_policy_server import MODEL, STEP_M, FakeServer

MS = 1_000_000


def rotz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


class WireTests(unittest.TestCase):
    def test_header_and_blobs_round_trip(self):
        a, b = socket.socketpair()
        with a, b:
            image = np.arange(256 * 256 * 3, dtype=np.uint32).astype(np.uint8).reshape(256, 256, 3)
            policy.send_msg(a, {"type": "x", "n": 1}, [image, b"\x01\x02"])
            header, blobs = policy.recv_msg(b)
        self.assertEqual(header, {"type": "x", "n": 1})
        np.testing.assert_array_equal(np.frombuffer(blobs[0], np.uint8).reshape(256, 256, 3), image)
        self.assertEqual(blobs[1], b"\x01\x02")

    def test_link_hello_and_infer(self):
        server = FakeServer()
        try:
            link = policy.Link(f"127.0.0.1:{server.port}", 5)
            self.assertEqual(link.model, MODEL)
            state = np.r_[.1, .2, .3, 1, 0, 0, 0, 1, 0, .4]
            images = [np.full((256, 256, 3), (10, 20, 30), np.uint8), np.zeros((256, 256, 3), np.uint8)]
            reply = link.infer(7, ["fixed", "wrist"], images, state, "t")
            link.close()
        finally:
            server.close()
        actions = np.asarray(reply["actions"])
        self.assertEqual(actions.shape, (10, 10))
        np.testing.assert_allclose(actions[:, 0], .1 + STEP_M * np.arange(10))
        self.assertLessEqual(reply["sent_ns"], reply["reply_ns"])
        request = server.requests[0]
        self.assertEqual((request["id"], request["instruction"], request["state"]), (7, "t", state.tolist()))
        self.assertEqual([i["camera"] for i in request["images"]], ["fixed", "wrist"])
        self.assertEqual(request["image_means"][0], [10., 20., 30.])  # the bytes arrive as sent

    def test_server_error_raises(self):
        a, b = socket.socketpair()
        link = policy.Link.__new__(policy.Link)
        link.sock = a
        policy.send_msg(b, {"type": "error", "id": 1, "error": "ValueError('image')"})
        with a, b, self.assertRaisesRegex(RuntimeError, "image"):
            link.infer(1, ["fixed"], [np.zeros((2, 2, 3), np.uint8)], np.zeros(10), "t")


class ObservationTests(unittest.TestCase):
    def test_crop_before_resize_and_rgb(self):
        # 640 x 480 BGR frame: the crop box (150..500, 100..450) is blue (BGR 255, 0, 0) with a red (0, 0, 255)
        # left half; outside it green. Cropped first, nothing green may remain, and the colours come out as RGB.
        image = np.zeros((480, 640, 3), np.uint8)
        image[:, :, 1] = 255
        image[100:450, 150:500] = (255, 0, 0)
        image[100:450, 150:325] = (0, 0, 255)
        out = policy.prepare(image, 256, (150, 100, 500, 450))
        self.assertEqual((out.shape, out.dtype), ((256, 256, 3), np.uint8))
        self.assertEqual(out[:, :, 1].max(), 0)  # no green: the outside was cut off before the resize
        np.testing.assert_array_equal(out[128, 10], (255, 0, 0))  # left half red, in RGB
        np.testing.assert_array_equal(out[128, 245], (0, 0, 255))  # right half blue, in RGB
        np.testing.assert_array_equal(out, vpp.square(image, 256, (150, 100, 500, 450))[:, :, ::-1])

    def test_whole_frame_without_crop(self):
        image = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        np.testing.assert_array_equal(policy.prepare(image, 256), vpp.square(image, 256)[:, :, ::-1])


class StateTests(unittest.TestCase):
    def test_interpolated_pose_and_last_gripper(self):
        t = np.array([0, 10 * MS, 20 * MS], np.int64)
        R = [rotz(0), rotz(.2), rotz(.4)]
        tcp = [np.r_[0.1 * k, 0, .5, r[:2].reshape(6)] for k, r in enumerate(R)]
        grip = [0., .5, 1.]
        s = policy.robot_state(t, tcp, grip, 15 * MS, 50 * MS)
        np.testing.assert_allclose(s[:3], [.15, 0, .5], atol=1e-9)
        np.testing.assert_allclose(s[3:9], rotz(.3)[:2].reshape(6), atol=1e-9)  # slerp halfway
        self.assertEqual(s[9], .5)  # the last record at that time, not interpolated
        self.assertIsNone(policy.robot_state(t, tcp, grip, 25 * MS, 50 * MS))  # after the last record
        self.assertIsNone(policy.robot_state(t, tcp, grip, 15 * MS, 5 * MS))  # records too far apart


class SettingsTests(unittest.TestCase):
    def test_mismatches(self):
        crop = {"fixed": (150, 100, 500, 450)}
        self.assertEqual(policy.mismatches(MODEL, ["fixed", "wrist"], crop, 256, 10), [])
        self.assertEqual(policy.mismatches(MODEL, ["fixed", "wrist"], {**crop, "other": (0, 0, 1, 1)}, 256, 10), [])
        problems = policy.mismatches(MODEL, ["wrist", "fixed"], {"fixed": (0, 0, 640, 480)}, 224, 10)
        self.assertEqual([p.split(":")[0] for p in problems], ["cameras", "crop", "size"])
        self.assertEqual(len(policy.mismatches(MODEL, ["fixed", "wrist"], {}, 256, 10)), 1)  # no crop: differs

    def test_replan_time(self):
        # Auto: chunk of 10 at 0.1 s, latency 0.35 s -> the next observation 0.55 s after the last one.
        self.assertEqual(policy.replan_at(1000 * MS, 10, 100 * MS, 0, 350 * MS), 1550 * MS)
        self.assertEqual(policy.replan_at(1000 * MS, 10, 100 * MS, 4, 350 * MS), 1400 * MS)  # REPLAN_STEPS=4

    def test_deploy_has_the_settings_of_the_teleop_dataflow(self):
        here = policy.HERE
        teleop_ur5, deploy_ur5 = policy.node_env(here / "dataflow.yml", "ur5"), policy.node_env(here / "deploy.yml", "ur5")
        self.assertGreater(len(teleop_ur5), 20)
        self.assertEqual(deploy_ur5.pop("CONTROL"), "policy")
        self.assertEqual(deploy_ur5, teleop_ur5)  # the robot as the demonstrations were recorded
        recorder, node = policy.node_env(here / "dataflow.yml", "recorder"), policy.node_env(here / "deploy.yml", "policy")
        for key in ("COLOR_W", "COLOR_H", "FPS", "MAX_POSE_GAP_MS", "VPP_CAMERAS", "VPP_CROP", "VPP_FPS", "VPP_SIZE"):
            self.assertEqual(node[key], recorder[key], key)


class FramesTests(unittest.TestCase):
    def test_latest_frame_copied_and_implausible_time_replaced(self):
        frames, seen = policy.Frames(), []
        frames.episode = type("E", (), {"add": lambda self, role, image, row: seen.append((role, row))})()
        image = np.zeros((4, 4, 3), np.uint8)
        frames.add("fixed", image, {"frame_mono_ns": 990 * MS, "host_receive_mono_ns": 1000 * MS, "frame_number": 1})
        image[:] = 9  # the SDK reuses its buffer
        self.assertEqual(frames.latest["fixed"][0].max(), 0)
        self.assertEqual(frames.latest["fixed"][1]["frame_mono_ns"], 990 * MS)
        # SDK time 3.9 s ahead (seen on the D455s): receive time - the typical latency (10 ms so far) instead.
        frames.add("fixed", image, {"frame_mono_ns": 5000 * MS, "host_receive_mono_ns": 1100 * MS, "frame_number": 2})
        self.assertEqual(frames.latest["fixed"][1]["frame_mono_ns"], 1090 * MS)
        self.assertEqual([row["frame_mono_ns"] for _, row in seen], [990 * MS, 5000 * MS])  # the episode: raw rows


if __name__ == "__main__":
    unittest.main()
