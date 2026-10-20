import os
import socket
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
import spacemouse
from spacemouse import daemon_alive


class DaemonAliveTests(unittest.TestCase):
    """spacenavd liveness without moving or restarting the real daemon: a socketpair stands in."""

    def test_alive_without_and_with_pending_data_dead_after_close(self):
        ours, daemon = socket.socketpair()
        try:
            self.assertTrue(daemon_alive(ours))        # connected, nothing to read
            daemon.sendall(b"x" * 32)
            self.assertTrue(daemon_alive(ours))        # an event is pending (peeked, not consumed)
            self.assertEqual(len(ours.recv(64)), 32)    # MSG_PEEK left the data in place
            daemon.close()
            self.assertFalse(daemon_alive(ours))       # peer gone: EOF
        finally:
            ours.close()


class Motion:
    def __init__(self, translation, rotation=(0, 0, 0)):
        self.translation, self.rotation = translation, rotation


class Button:
    def __init__(self, bnum, press):
        self.bnum, self.press = bnum, press


class NodeTests(unittest.TestCase):
    def run_node(self, ticks, **extra_env):
        """ticks: per Dora tick, the spnav samples pending; "dead" = the daemon exits before that tick."""
        ours, daemon = socket.socketpair()
        pending = []
        fake = types.SimpleNamespace(SpnavMotionEvent=Motion, SpnavButtonEvent=Button, spnav_open=MagicMock(),
                                     spnav_close=MagicMock(), spnav_fd=lambda: ours.fileno(),
                                     spnav_poll_event=lambda: pending.pop(0) if pending else None)

        def events():
            for samples in ticks:
                if samples == "dead":
                    daemon.close()
                    samples = []
                pending[:] = samples
                yield {"type": "INPUT", "id": "tick"}
            yield {"type": "STOP"}
        env = {"MOCK_MOUSE": "0", "MOUSE_MAX": "250", "DEADZONE": "0.24", **extra_env}
        try:
            with patch.dict(sys.modules, {"spnav": fake}), patch.dict(os.environ, env), \
                    patch.object(spacemouse, "Node", lambda: events()), \
                    patch.object(spacemouse, "send_json") as sent:
                try:
                    spacemouse.main()
                finally:
                    self.sent = [(c.args[1], c.args[2]) for c in sent.call_args_list]
                    fake.spnav_close.assert_called_once()
        finally:
            ours.close()
            daemon.close()
        return [payload["axes"] for name, payload in self.sent if name == "command"]

    def test_held_cap_keeps_last_deflection_until_release(self):
        # raw y 155 of MOUSE_MAX 250, deadzone .24 -> +0.5 on the mapped z axis; then 30 ticks without events.
        axes = self.run_node([[Motion((0, 155, 0))], *[[]] * 30, [Motion((0, 20, 0))], [], []])
        self.assertEqual(len(axes), 34)
        self.assertTrue(all(a == [0, 0, .5, 0, 0, 0] for a in axes[:31]))  # held, never timed out
        self.assertTrue(all(a == [0] * 6 for a in axes[31:]))               # release: back in the deadzone
        self.assertTrue(all(p["enabled"] for name, p in self.sent if name == "command"))

    def test_rotation_shaping_from_env(self):
        # raw rx 162.5 of 250: deadzone .3 and square curve -> .25; with a .5 push on y, coupling 1 hides it.
        axes = self.run_node([[Motion((0, 0, 0), (162.5, 0, 0))], [Motion((0, 155, 0), (162.5, 0, 0))]],
                             ROTATION_DEADZONE="0.3", ROTATION_CURVE="2", ROTATION_COUPLING="1")
        self.assertEqual([[round(v, 12) for v in a] for a in axes], [[0, 0, 0, 0, .25, 0], [0, 0, .5, 0, 0, 0]])

    def test_dead_daemon_ends_the_node(self):
        with self.assertRaisesRegex(RuntimeError, "spacenavd connection lost"):
            self.run_node([[Motion((0, 155, 0))], [], "dead", []])
        self.assertEqual(len([name for name, _ in self.sent if name == "command"]), 2)  # nothing after it died

    def test_button_press_edges(self):
        self.run_node([[Button(0, True)], [Button(0, False)], [Button(1, True), Button(1, False)], [Button(0, True)]])
        self.assertEqual([p["button"] for name, p in self.sent if name == "gripper"], ["open", "close", "open"])


if __name__ == "__main__":
    unittest.main()
