import os
import unittest
from unittest.mock import patch
import numpy as np
from common import (clamp_box, flange_from_tcp, follow_leash, integrate_pose, limited_twist, map_motion,
                    mat_to_rot6d, matrix_pose, pose9d, pose_error, pose_matrix, rot6d_to_mat, smooth_twist, vector,
                    workspace_bounds, within_workspace)


class TeleopTests(unittest.TestCase):
    def test_reference_axis_mapping(self):
        np.testing.assert_allclose(map_motion([30, 60, 90, 120, 150, 180], deadzone=0),
                                   [-.3, .1, .2, -.6, .4, .5])

    def test_deadzone_and_clip(self):
        np.testing.assert_array_equal(map_motion([1, 0, 900, 0, 0, 0]), [-1, 0, 0, 0, 0, 0])

    def test_deadzone_rescaled_without_step(self):
        # MOUSE_MAX 250, DEADZONE 0.24: raw 60 is the edge (0), raw 250 is full scale.
        x = [map_motion([0, raw, 0, 0, 0, 0], 250, .24)[2] for raw in (59, 60, 61, 155, 250, 350)]
        np.testing.assert_allclose(x, [0, 0, 1 / 190, .5, 1, 1], atol=1e-12)
        self.assertAlmostEqual(map_motion([0, -155, 0, 0, 0, 0], 250, .24)[2], -.5)

    def test_smooth_twist(self):
        target, twist = np.r_[.08, 0, 0, 0, 0, .2], np.zeros(6)
        for _ in range(20):  # 20 steps of tau/20 = one time constant
            twist = smooth_twist(twist, target, .15 / 20, .15)
        np.testing.assert_allclose(twist, target * (1 - np.exp(-1)), rtol=1e-6)

    def test_release_stale_and_future_stop(self):
        command = {"t_ns": 1_000_000_000, "enabled": True, "axes": [1.] * 6}
        for now, enabled in [(1_000_000_000, False), (1_200_000_000, True), (999_999_999, True)]:
            np.testing.assert_array_equal(limited_twist({**command, "enabled": enabled}, now, .15, .03, .15), np.zeros(6))
        np.testing.assert_array_equal(limited_twist(None, 0, .15, .03, .15), np.zeros(6))

    def test_vector_speed_limits(self):
        command = {"t_ns": 1, "enabled": True, "axes": [1.] * 6}
        speed = limited_twist(command, 2, .15, .03, .15)
        self.assertAlmostEqual(np.linalg.norm(speed[:3]), .03)
        self.assertAlmostEqual(np.linalg.norm(speed[3:]), .15)

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            limited_twist({"t_ns": 1, "enabled": True, "axes": [float("nan")] * 6}, 2, .1, .03, .15)

    def test_workspace(self):
        lo, hi = np.zeros(3), np.ones(3)
        self.assertTrue(within_workspace([.5, .5, .5, 0, 0, 0], np.zeros(6), lo, hi, .02))
        self.assertFalse(within_workspace([.975, .5, .5, 0, 0, 0], [.03, 0, 0, 0, 0, 0], lo, hi, .02))
        empty = {"WORKSPACE_MIN": "", "WORKSPACE_MAX": "", "WORKSPACE_OFFSET_MIN": "", "WORKSPACE_OFFSET_MAX": "",
                 "WORKSPACE_Z_MIN": ""}
        with patch.dict(os.environ, empty):
            lo, hi = workspace_bounds(real=True, start_xyz=[.5, .5, .5])
            np.testing.assert_allclose(lo, [.35, .35, .45])
            np.testing.assert_allclose(hi, [.65, .65, .65])
        with patch.dict(os.environ, {**empty, "WORKSPACE_MIN": "[0, 0, 0]"}):
            with self.assertRaises(ValueError):  # only one absolute bound given
                workspace_bounds(real=True, start_xyz=[.5, .5, .5])

    def test_absolute_floor(self):
        env = {"WORKSPACE_MIN": "", "WORKSPACE_MAX": "", "WORKSPACE_OFFSET_MIN": "", "WORKSPACE_OFFSET_MAX": ""}
        home = [-0.4878, -0.1101, 0.2469]  # TCP at home, 2026-09-24 run
        twist = [-0.0059, 0, -0.0081, 0, 0, 0]
        stopped_at = [-0.514, -0.1099, 0.2189, 0, 0, 0]  # where that run hit the old 5 cm floor
        with patch.dict(os.environ, {**env, "WORKSPACE_Z_MIN": "-0.0481"}):
            lo, hi = workspace_bounds(real=True, start_xyz=home)
            np.testing.assert_allclose(lo, [-0.6378, -0.2601, -0.0481])
            np.testing.assert_allclose(hi, [-0.3378, 0.0399, 0.3969])
            self.assertTrue(within_workspace(stopped_at, twist, lo, hi, .02))
            self.assertTrue(within_workspace([-0.5, -0.1, -0.027, 0, 0, 0], np.zeros(6), lo, hi, .02))
            self.assertFalse(within_workspace([-0.5, -0.1, -0.029, 0, 0, 0], np.zeros(6), lo, hi, .02))  # floor + margin
        with patch.dict(os.environ, {**env, "WORKSPACE_Z_MIN": "0.5"}):
            with self.assertRaises(ValueError):  # floor above the top of the box
                workspace_bounds(real=True, start_xyz=home)

    def test_null_axis_is_unlimited(self):
        env = {"WORKSPACE_MIN": "", "WORKSPACE_MAX": "", "WORKSPACE_OFFSET_MIN": "[null, null, -0.05]",
               "WORKSPACE_OFFSET_MAX": "[null, null, 0.15]", "WORKSPACE_Z_MIN": "-0.0481"}
        with patch.dict(os.environ, env):
            lo, hi = workspace_bounds(real=True, start_xyz=[-0.4878, -0.1101, 0.2469])
        np.testing.assert_allclose(lo, [-np.inf, -np.inf, -0.0481])
        np.testing.assert_allclose(hi, [np.inf, np.inf, 0.3969])
        self.assertTrue(within_workspace([-0.666, -0.022, -0.02, 0, 0, 0], np.zeros(6), lo, hi, .02))  # recorded floor x/y
        self.assertTrue(within_workspace([-2.0, 3.0, 0.2, 0, 0, 0], [0.01, 0, 0, 0, 0, 0], lo, hi, .02))
        self.assertFalse(within_workspace([-0.5, -0.1, -0.029, 0, 0, 0], np.zeros(6), lo, hi, .02))  # floor
        self.assertFalse(within_workspace([-0.5, -0.1, 0.378, 0, 0, 0], np.zeros(6), lo, hi, .02))  # ceiling
        with self.assertRaises(ValueError):  # null is only for workspace bounds, not joint targets
            vector("HOME_Q_DEG", "[0, -90, 90, -90, null, 180]", 6)


class FlangeTests(unittest.TestCase):
    def test_pose_matrix_round_trip_including_near_pi(self):
        rng = np.random.default_rng(0)
        poses = [np.r_[rng.normal(size=3), rng.normal(size=3)] for _ in range(20)]
        poses += [np.array([-0.4878, -0.1101, 0.2469, 2.2302, 2.2111, 0.0071]),  # home, angle ~178.6 deg
                  np.array([0.1, 0.2, 0.3, np.pi, 0, 0]), np.zeros(6)]
        for pose in poses:
            T = pose_matrix(pose)
            np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(pose_matrix(matrix_pose(T)), T, atol=1e-9)

    def test_flange_from_tcp_known_answer(self):
        # Tool pointing down (180 deg about x), TCP 0.15 m along tool z: flange sits 0.15 m above the TCP.
        flange = flange_from_tcp([0.5, 0.0, 0.2, np.pi, 0, 0], [0, 0, 0.15, 0, 0, 0])
        np.testing.assert_allclose(flange, pose_matrix([0.5, 0.0, 0.35, np.pi, 0, 0]), atol=1e-12)
        np.testing.assert_allclose(flange_from_tcp([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0] * 6),
                                   pose_matrix([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), atol=1e-12)

    def test_rot6d_is_first_two_rows_and_round_trips(self):
        # Same convention as UMI pose_util.mat_to_rot6d / pytorch3d matrix_to_rotation_6d: mat[..., :2, :].
        rng = np.random.default_rng(2)
        for v in list(rng.normal(size=(50, 3))) + [np.array([np.pi - 1e-6, 0, 0])]:
            R = pose_matrix(np.r_[0, 0, 0, v])[:3, :3]
            np.testing.assert_array_equal(mat_to_rot6d(R), R[:2, :].reshape(6))
            np.testing.assert_allclose(rot6d_to_mat(mat_to_rot6d(R)), R, atol=1e-12)
        T = pose_matrix([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        np.testing.assert_allclose(pose9d(T), np.r_[0.1, 0.2, 0.3, T[0, :3], T[1, :3]])


class TargetTests(unittest.TestCase):
    """Option B (DP/UMI style): integrate the mouse twist into a target pose for servoL."""

    def test_integrate_translation_and_base_frame_rotation(self):
        T0 = pose_matrix([0.4, -0.1, 0.25, 2.2302, 2.2111, 0.0071])  # home-like orientation, ~178.6 deg
        T = T0
        for _ in range(125):  # 1 s of [10 mm/s x, 0.2 rad/s about base z]
            T = integrate_pose(T, [0.01, 0, 0, 0, 0, 0.2], 1 / 125)
        np.testing.assert_allclose(T[:3, 3], T0[:3, 3] + [0.01, 0, 0], atol=1e-12)   # rotation is about the TCP
        np.testing.assert_allclose(T[:3, :3], pose_matrix([0, 0, 0, 0, 0, 0.2])[:3, :3] @ T0[:3, :3], atol=1e-12)
        np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-12)  # stays a rotation

    def test_clamp_box_keeps_margin_and_ignores_unlimited_axes(self):
        lo, hi = np.array([-np.inf, -np.inf, -0.0965]), np.array([np.inf, np.inf, 0.358])
        T = clamp_box(pose_matrix([5.0, -7.0, -0.2, 0, 0, 0]), lo, hi, 0.02)
        np.testing.assert_allclose(T[:3, 3], [5.0, -7.0, -0.0765])
        T = clamp_box(pose_matrix([0, 0, 0.5, 0, 0, 0]), lo, hi, 0.02)
        self.assertAlmostEqual(T[2, 3], 0.338)

    def test_clamp_box_never_snaps_a_target_inside_the_margin(self):
        lo, hi = np.zeros(3), np.ones(3)
        prev = pose_matrix([0.99, 0.5, 0.5, 0, 0, 0])  # e.g. started 1 cm from the wall, margin 2 cm
        out = clamp_box(pose_matrix([0.995, 0.5, 0.5, 0, 0, 0]), lo, hi, 0.02, prev)
        self.assertAlmostEqual(out[0, 3], 0.99)       # may not go further out ...
        out = clamp_box(pose_matrix([0.985, 0.5, 0.5, 0, 0, 0]), lo, hi, 0.02, prev)
        self.assertAlmostEqual(out[0, 3], 0.985)      # ... moves inward freely, no jump to 0.98
        out = clamp_box(pose_matrix([0.99, 0.5, 0.5, 0, 0, 0]), lo, hi, 0.02, out)
        self.assertAlmostEqual(out[0, 3], 0.985)      # and does not drift back out
        inside = pose_matrix([0.5, 0.5, 0.5, 0, 0, 0])
        np.testing.assert_allclose(clamp_box(pose_matrix([1.5, 0.5, 0.5, 0, 0, 0]), lo, hi, 0.02, inside)[:3, 3],
                                   [0.98, 0.5, 0.5])  # normal case unchanged

    def test_leash_stops_runaway_target_but_allows_return(self):
        actual = pose_matrix([0, 0, 0, 0, 0, 0])  # robot not following
        T = actual
        for _ in range(100):
            T = follow_leash(T, integrate_pose(T, [0.1, 0, 0, 0, 0, 0], 0.008), actual, 0.03, 0.2)
        self.assertLessEqual(pose_error(T, actual)[0], 0.03 + 1e-12)
        self.assertGreater(pose_error(T, actual)[0], 0.029)
        back = follow_leash(T, integrate_pose(T, [-0.1, 0, 0, 0, 0, 0], 0.008), actual, 0.03, 0.2)
        self.assertLess(pose_error(back, actual)[0], pose_error(T, actual)[0])  # moving back is allowed
        T = actual
        for _ in range(200):
            T = follow_leash(T, integrate_pose(T, [0, 0, 0, 0, 0, 0.5], 0.008), actual, 0.03, 0.2)
        self.assertLessEqual(pose_error(T, actual)[1], 0.2 + 1e-9)


if __name__ == "__main__":
    unittest.main()
