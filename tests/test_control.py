"""Exercise the real-mode control path using fake RTDE interfaces, no network."""
import importlib.util
import itertools
import math
import os
import signal
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import MagicMock, patch
import numpy as np  # also loaded before patch.dict(sys.modules) so it is never unloaded


HOME_DEG = [0, -90, 90, -90, -88, 180]
HOME = [math.radians(d) for d in HOME_DEG]


def load_ur5(node=None):
    spec = importlib.util.spec_from_file_location("tested_ur5", Path(__file__).resolve().parents[1] / "ur5.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"dora": types.SimpleNamespace(Node=lambda: node)}):
        spec.loader.exec_module(module)
    return module


class HomeTests(unittest.TestCase):
    def fakes(self, q):
        control, receive = MagicMock(), MagicMock()
        control.isJointsWithinSafetyLimits.return_value = True
        control.moveJ.return_value = True
        receive.getRobotMode.return_value, receive.getSafetyMode.return_value = 7, 1
        receive.getActualQ.return_value = q
        receive.getActualQd.return_value = [0.] * 6
        receive.getActualTCPPose.return_value = [.5, .5, .5, 0., 0., 0.]
        return control, receive

    def test_reaches_home_with_async_movej(self):
        module, node = load_ur5(), MagicMock()
        node.next.return_value = None  # no STOP
        control, receive = self.fakes(HOME)
        with patch.object(module, "print", create=True):
            self.assertTrue(module.move_home(node, control, receive, module.np.radians(HOME_DEG), .2, .3, .01))
        args = control.moveJ.call_args.args
        self.assertEqual((args[1], args[2], args[3]), (.2, .3, True))
        module.np.testing.assert_allclose(args[0], HOME)
        control.stopJ.assert_not_called()

    def test_esc_homing_kicks_watchdog(self):
        module, node = load_ur5(), MagicMock()
        node.next.return_value = None
        control, receive = self.fakes(HOME)
        receive.getActualQ.side_effect = [[0.] * 6, [0.] * 6, HOME, HOME]  # arrives on the 2nd poll
        with patch.object(module, "print", create=True):
            self.assertTrue(module.move_home(node, control, receive, module.np.radians(HOME_DEG), .2, .3, 0,
                                             watchdog=True))
        control.kickWatchdog.assert_called()

    def test_input_closed_does_not_abort_homing(self):
        module, node = load_ur5(), MagicMock()
        closed = iter([{"type": "INPUT_CLOSED", "id": "esc"}])
        node.next.side_effect = lambda **kw: next(closed, None)
        self.assertFalse(module.stop_requested(node, 0.05))

    def test_inputs_dropped_while_homing_are_collected(self):
        module, node = load_ur5(), MagicMock()
        events = iter([{"type": "INPUT", "id": "esc"}, {"type": "INPUT_CLOSED", "id": "esc"},
                       {"type": "INPUT", "id": "heartbeat"}])
        node.next.side_effect = lambda **kw: next(events, None)
        seen = set()
        self.assertFalse(module.stop_requested(node, 0.05, seen))
        self.assertEqual(seen, {"esc", "heartbeat"})

    def test_open_gripper_waits_until_stopped_and_kicks_watchdog(self):
        module, node = load_ur5(), MagicMock()
        node.next.return_value = None
        control, gripper = MagicMock(), MagicMock(error=None)
        gripper.snapshot.side_effect = [{"mode": "opening"}] * 3 + [{"mode": "stopped"}]
        with patch.object(module, "print", create=True):
            self.assertTrue(module.open_gripper(node, control, gripper, True))
        gripper.open.assert_called_once()
        self.assertEqual(control.kickWatchdog.call_count, 3)  # once per wait while opening

    def test_open_gripper_timeout_warns_and_stop_aborts(self):
        module, node = load_ur5(), MagicMock()
        node.next.return_value = None
        gripper = MagicMock(error=None)
        gripper.snapshot.return_value = {"mode": "opening"}  # never reports stopped
        with patch.object(module, "print", create=True) as printed:
            self.assertTrue(module.open_gripper(node, None, gripper, False, timeout=0.05))  # homes anyway
        self.assertTrue(any("WARNING" in c.args[0] for c in printed.call_args_list))
        node.next.return_value = {"type": "STOP"}
        with patch.object(module, "print", create=True):
            self.assertFalse(module.open_gripper(node, None, gripper, False))
        gripper.error = "socket closed"
        with self.assertRaisesRegex(RuntimeError, "Gripper failed"), patch.object(module, "print", create=True):
            module.open_gripper(node, None, gripper, False)

    def test_stop_during_countdown_skips_motion(self):
        module, node = load_ur5(), MagicMock()
        node.next.return_value = {"type": "STOP"}
        control, receive = self.fakes([0.] * 6)
        with patch.object(module, "print", create=True):
            self.assertFalse(module.move_home(node, control, receive, module.np.radians(HOME_DEG), .2, .3, .01))
        control.moveJ.assert_not_called()

    def test_stop_during_motion_calls_stopj(self):
        module, node = load_ur5(), MagicMock()
        control, receive = self.fakes([0.] * 6)  # never arrives
        node.next.side_effect = lambda **kw: {"type": "STOP"} if control.moveJ.called else None
        with patch.object(module, "print", create=True):
            self.assertFalse(module.move_home(node, control, receive, module.np.radians(HOME_DEG), .2, .3, .01))
        control.stopJ.assert_called()


class PacketTests(unittest.TestCase):
    """The controller's RTDE packets are the loop clock (real mode); dry run uses absolute deadlines."""

    def test_waits_for_a_newer_controller_packet(self):
        module, receive = load_ur5(), MagicMock()
        receive.getTimestamp.side_effect = [1.0, 1.0, 1.0, 1.008]  # receive thread has no new packet yet
        self.assertEqual(module.wait_for_packet(receive, 1.0, 1.0), 1.008)
        self.assertEqual(receive.getTimestamp.call_count, 4)

    def test_stalled_controller_raises(self):
        module, receive = load_ur5(), MagicMock()
        receive.getTimestamp.return_value = 1.0
        t0 = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "stale"):
            module.wait_for_packet(receive, 1.0, .02)
        self.assertLess(time.monotonic() - t0, .2)

    def test_dry_run_deadlines_are_absolute(self):
        module = load_ur5()
        self.assertAlmostEqual(module.next_tick(0., .008, .001), .008)  # normal: next deadline, overhead not added
        self.assertAlmostEqual(module.next_tick(0., .008, .012), .008)  # slightly late: catch up, no drift
        self.assertAlmostEqual(module.next_tick(0., .008, .5), .5)      # far behind: resync instead of bursting


class ControlTests(unittest.TestCase):
    def run_control(self, fail_servo=False, auto_box=False, with_gripper=False, home=False, esc=False,
                    release=False, start=(.5, .5, .5), hold=0, env_extra=None, expect_error=None, stamps=None,
                    reset=False, esc_while_homing=False, h_key=False, recording=True):
        control = MagicMock()
        receive = MagicMock()
        receive.getRobotMode.return_value = 7
        receive.getSafetyMode.return_value = 1
        receive.getActualTCPPose.return_value = [*start, 0., 0., 0.]
        # Controller timestamps as polled by the loop; default: a new packet at every poll.
        stamps = iter(stamps if stamps is not None else (1 + .008 * i for i in itertools.count()))
        last = [None]
        receive.getTimestamp.side_effect = lambda: last.__setitem__(0, next(stamps, last[0])) or last[0]
        control.servoL.return_value = not fail_servo
        control.getTCPOffset.return_value = [0., 0., 0.15, 0., 0., 0.]
        receive.getActualTCPSpeed.return_value = [0.] * 6
        # release: cap back inside the deadzone (enabled, zero axes) -> ramp down, no hard stop.
        released = [{"type": "INPUT", "id": "command", "payload": {"t_ns": time.monotonic_ns(), "enabled": True, "axes": [0] * 6}},
                    *[None] * 120] if release else []
        events = iter([
            {"type": "INPUT", "id": "command", "payload": {"t_ns": time.monotonic_ns(), "enabled": True, "axes": [1, 0, 0, 0, 0, 0]}},
            {"type": "INPUT", "id": "heartbeat", "payload": {"t_ns": time.monotonic_ns(), "recording": recording}},
            {"type": "INPUT", "id": "gripper", "payload": {"t_ns": time.monotonic_ns(), "button": "close"}},
            None,
            *[None] * hold,  # keep pushing for `hold` more cycles
            *released,
            {"type": "INPUT", "id": "command", "payload": {"t_ns": time.monotonic_ns(), "enabled": False, "axes": [1, 0, 0, 0, 0, 0]}},
            None,
            *([{"type": "INPUT", "id": "reset", "payload": {"id": 7}}, None, None] if reset else []),  # recorder
            *([{"type": "INPUT", "id": "home"}, None, None] if h_key else []),  # h key
            {"type": "INPUT", "id": "esc"} if esc else {"type": "STOP"},
            {"type": "INPUT_CLOSED", "id": "esc"},  # keyboard node exits after Esc
        ])
        node = MagicMock()
        node.next.side_effect = lambda **kw: next(events, None)  # exhausted = dora timeout
        spec = importlib.util.spec_from_file_location("tested_ur5", Path(__file__).resolve().parents[1] / "ur5.py")
        module = importlib.util.module_from_spec(spec)
        modules = {"dora": types.SimpleNamespace(Node=lambda: node),
                   "rtde_control": types.SimpleNamespace(RTDEControlInterface=lambda *a: control),
                   "rtde_receive": types.SimpleNamespace(RTDEReceiveInterface=lambda *a: receive)}
        env = {"DRY_RUN": "0", "ROBOT_IP": "fake", "WORKSPACE_MIN": "[0,0,0]", "WORKSPACE_MAX": "[1,1,1]",
               "GRIPPER": "1" if with_gripper else "0", "HOME_Q_DEG": str(HOME_DEG) if home else "",
               **(env_extra or {})}
        self.order = []  # gripper open / arm moveJ, in call order
        if home:
            control.isJointsWithinSafetyLimits.return_value = True
            control.moveJ.side_effect = lambda *a: self.order.append("moveJ") or True
            receive.getActualQ.return_value = HOME
            receive.getActualQd.return_value = [0.] * 6
        gripper = MagicMock(error=None)
        gripper.snapshot.return_value = {"mode": "closing", "pos": 10, "obj": 0, "t_ns": 1}

        def opened():  # the firmware reports fully open at once
            self.order.append("open")
            gripper.snapshot.return_value = {"mode": "stopped", "pos": 0, "obj": 3, "t_ns": 2}
        gripper.open.side_effect = opened
        if auto_box:  # box from the start TCP read over RTDE
            env.update(WORKSPACE_MIN="", WORKSPACE_MAX="", WORKSPACE_OFFSET_MIN="", WORKSPACE_OFFSET_MAX="")
        printed = MagicMock()
        with patch.dict(sys.modules, modules), patch.dict("os.environ", env):
            spec.loader.exec_module(module)
            with patch.object(module, "send_json") as sent, patch.object(module, "read_json", lambda e: e["payload"]), \
                    patch.object(module, "print", printed, create=True), \
                    patch.object(module, "Gripper", lambda *a, **k: gripper), \
                    patch.object(module, "stop_requested", lambda node, seconds, inputs=None:
                                 esc_while_homing and inputs is not None and inputs.add("esc")), \
                    patch.object(module.os, "kill") as kill:
                if fail_servo or expect_error:
                    with self.assertRaisesRegex(RuntimeError, expect_error or "servoL failed"):
                        module.main()
                else:
                    module.main()
        # Option B: no speed streaming and no blocking stop inside the loop.
        control.speedL.assert_not_called()
        control.speedStop.assert_not_called()
        # Paced by the controller's packets, not by the host timer.
        control.initPeriod.assert_not_called()
        control.waitPeriod.assert_not_called()
        self.servo_x = [c.args[0][0] for c in control.servoL.call_args_list]
        if not expect_error and not recording:  # no episode recorded: the mouse push never moves the target
            self.assertTrue(all(x == start[0] for x in self.servo_x))
        elif not expect_error:
            tau = float((env_extra or {}).get("TWIST_FILTER_S", .15))
            first = .03 * (1 - math.exp(-(1 / 125) / tau))  # default MAX_LINEAR_M_S, low-passed one cycle
            edge = max(1 - .02, start[0])  # box [0, 1], margin .02; a start inside the margin is held, not snapped
            self.assertAlmostEqual(self.servo_x[0], min(start[0] + first / 125, edge))  # target integrates the twist
            args = control.servoL.call_args_list[0].args
            self.assertEqual((args[3], args[4], args[5]), (1 / 125, .1, 300))  # dt, lookahead, gain
            if release:
                steps = np.diff([start[0], *self.servo_x])
                moving = steps[steps > 0]
                self.assertGreater(len(moving), 10)
                self.assertTrue(all(b < a for a, b in zip(moving, moving[1:])))  # increments shrink: ramp down
            elif not fail_servo and not hold and not (reset or h_key):  # homing: the target restarts at home
                self.assertTrue(all(x == self.servo_x[0] for x in self.servo_x))  # disabled -> target frozen
        if control.servoL.called:  # STOP_ACCELERATION, only at shutdown (and before a reset moveJ)
            self.assertEqual([c.args for c in control.servoStop.call_args_list], [(1.,)] * (2 if (reset or h_key) and not esc_while_homing else 1))
        control.stopScript.assert_called_once()
        control.disconnect.assert_called_once()
        receive.disconnect.assert_called_once()
        lines = [c.args[0] for c in printed.call_args_list]
        self.assertTrue(any(line.startswith("REAL mouse=") for line in lines))
        self.states = [c.args[2] for c in sent.call_args_list if c.args[1] == "state"]
        self.gripper, self.control, self.kill = gripper, control, kill
        return lines

    def test_motion_release_and_shutdown(self):
        self.run_control()

    def test_deadzone_release_ramps_down(self):
        self.run_control(release=True)

    def test_failed_motion_still_stops(self):
        self.run_control(fail_servo=True)

    def test_not_recording_freezes_arm_and_ignores_gripper(self):
        lines = self.run_control(with_gripper=True, recording=False)
        self.gripper.press.assert_not_called()
        self.assertIn("gripper close button ignored: not recording", lines)
        self.assertTrue(all(s["hard_stop"] and s["recording"] is False for s in self.states))
        self.assertTrue(all(s["gripper"]["events"] == [] for s in self.states))

    def test_gripper_action_in_same_state_record(self):
        self.run_control(with_gripper=True)
        self.gripper.start.assert_called_once()
        self.gripper.press.assert_called_once_with("close")
        self.gripper.close.assert_called_once()
        # The press is logged in the same record as the arm pose and twist of that cycle.
        first = self.states[0]
        self.assertEqual(first["gripper"]["events"], ["close"])
        self.assertEqual(first["gripper"]["mode"], "closing")
        self.assertIn("tcp_pose", first)
        self.assertEqual(self.states[1]["gripper"]["events"], [])

    def test_startup_opens_gripper_then_homes_then_watchdog(self):
        self.run_control(with_gripper=True, home=True)
        self.assertEqual(self.order, ["open", "moveJ"])  # episode start state: gripper open, arm home
        calls = [c[0] for c in self.control.mock_calls]
        self.assertLess(calls.index("moveJ"), calls.index("setWatchdog"))  # moveJ does not kick the watchdog
        self.assertEqual([c.args[0] for c in self.gripper.press.call_args_list], ["close"])  # the button press only

    def test_recorder_reset_opens_gripper_then_returns_home(self):
        self.run_control(with_gripper=True, home=True, reset=True)
        self.assertEqual(self.order, ["open", "moveJ", "open", "moveJ"])  # startup, then the reset

    def test_h_returns_home_without_touching_the_gripper(self):
        lines = self.run_control(with_gripper=True, home=True, h_key=True)
        self.assertIn("h: returning home", lines)
        self.assertEqual(self.order, ["open", "moveJ", "moveJ"])  # startup open only: h may be carrying an object

    def test_esc_returns_home_then_stops_dataflow(self):
        self.run_control(home=True, esc=True)
        self.assertEqual(self.control.moveJ.call_count, 2)  # startup homing + Esc homing
        calls = [c[0] for c in self.control.mock_calls]
        last_move = len(calls) - 1 - calls[::-1].index("moveJ")
        self.assertLess(calls.index("servoStop"), last_move)  # leave servo mode before the Esc moveJ
        self.kill.assert_called_once_with(os.getppid(), signal.SIGINT)

    def test_recorder_reset_returns_home_and_reports_done(self):
        lines = self.run_control(home=True, reset=True)
        self.assertEqual(self.control.moveJ.call_count, 2)  # startup homing + reset
        self.assertIn("reset: returning home", lines)
        calls = [c[0] for c in self.control.mock_calls]
        last_move = len(calls) - 1 - calls[::-1].index("moveJ")
        self.assertLess(calls.index("servoStop"), last_move)  # leave servo mode before the reset moveJ
        done = [s["reset_done"] for s in self.states]
        self.assertEqual((done[0], done[-1]), (0, 7))  # the recorder sees the reset id once home is reached
        self.assertEqual(done, sorted(done))
        self.kill.assert_not_called()  # teleop goes on

    def test_esc_during_reset_homing_still_homes_and_quits(self):
        self.run_control(home=True, reset=True, esc_while_homing=True)
        self.assertEqual(self.control.moveJ.call_count, 3)  # startup + reset + Esc
        self.kill.assert_called_once_with(os.getppid(), signal.SIGINT)
        self.assertEqual(self.states[-1]["reset_done"], 0)  # quit right after homing: no state record any more

    def test_stop_exits_without_homing(self):
        self.run_control(home=True)
        self.assertEqual(self.control.moveJ.call_count, 1)  # startup homing only
        self.kill.assert_not_called()

    def test_state_has_flange_pose_from_tcp_offset(self):
        self.run_control()
        self.control.getTCPOffset.assert_called_once()
        state = self.states[0]
        # Every recorded pose is xyz + rot6d (first two rows of R): identity -> [1, 0, 0, 0, 1, 0].
        np.testing.assert_allclose(state["tcp_offset"], [0., 0., 0.15, 1., 0., 0., 0., 1., 0.], atol=1e-12)
        np.testing.assert_allclose(state["tcp_pose"], [.5, .5, .5, 1., 0., 0., 0., 1., 0.], atol=1e-12)
        self.assertEqual(len(state["target_pose"]), 9)  # commanded target, xyz + rot6d
        self.assertFalse(state["hard_stop"])
        self.assertAlmostEqual(state["target_pose"][0], self.servo_x[0])
        for key in ("tcp_speed", "command_twist", "target_twist"):  # velocities stay 6-D (m/s, rad/s)
            self.assertEqual(len(state[key]), 6, key)
        # TCP [.5, .5, .5] with identity rotation; offset 0.15 along tool z -> flange 0.15 lower in z.
        # flange_pose = xyz + rot6d (first two rows of R); identity rotation -> [1, 0, 0, 0, 1, 0].
        np.testing.assert_allclose(state["flange_pose"], [.5, .5, .35, 1., 0., 0., 0., 1., 0.], atol=1e-12)

    def test_target_clamped_at_workspace_edge(self):
        self.run_control(start=(.985, .5, .5))  # box [0, 1], margin .02: pushing +x must not move it out
        self.assertTrue(all(x == .985 for x in self.servo_x))  # nor snap it in to .98 (no target jump)

    def test_actual_outside_workspace_raises(self):
        self.run_control(start=(1.01, .5, .5), expect_error="outside")

    def test_leash_when_robot_does_not_follow(self):
        # The fake robot never moves; after ~2 s of pushing the target must stay within 3 cm of it.
        self.run_control(hold=300, env_extra={"COMMAND_TIMEOUT_S": "100"})
        self.assertLessEqual(max(self.servo_x) - .5, .03 + 1e-9)
        self.assertGreater(max(self.servo_x) - .5, .025)

    def test_one_servo_command_per_controller_packet(self):
        # Each packet is seen by 3 polls before the next arrives: still exactly one servoL per packet.
        self.run_control(hold=5, stamps=(1 + .008 * (i // 3) for i in itertools.count()))
        packets = [s["robot_timestamp_s"] for s in self.states]
        self.assertTrue(all(b > a for a, b in zip(packets, packets[1:])))  # no duplicate rows
        self.assertEqual(self.control.servoL.call_count, len(self.states))

    def test_missed_packet_integrates_controller_time(self):
        # Packet 1.024 is missed (loop overran): that cycle integrates 16 ms, so the target keeps pace.
        stamps = [1.0, 1.008, 1.016, 1.032] + [1.032 + .008 * k for k in range(1, 1000)]
        self.run_control(hold=5, stamps=stamps, env_extra={"TWIST_FILTER_S": "1e-6"})  # no low-pass ramp
        steps = np.diff([.5, *self.servo_x[:5]])
        np.testing.assert_allclose(steps, .03 * np.array([.008, .008, .016, .008, .008]), rtol=1e-6)

    def test_stalled_feedback_raises(self):
        # Three packets, then the controller goes silent: stop with an error (and leave servo mode).
        self.run_control(hold=5, stamps=[1.0, 1.008, 1.016, 1.024], expect_error="stale")
        self.assertEqual(self.control.servoL.call_count, 3)

    def test_auto_workspace_from_start_tcp(self):
        lines = self.run_control(auto_box=True)
        workspace = next(line for line in lines if line.startswith("TCP workspace"))
        self.assertIn("min [0.35, 0.35, 0.45] max [0.65, 0.65, 0.65]", workspace)


if __name__ == "__main__":
    unittest.main()
