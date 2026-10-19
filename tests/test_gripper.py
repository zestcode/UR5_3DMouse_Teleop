import time
import unittest
from gripper import FakeRobotiq, Gripper


def wait_for(predicate, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class GripperTests(unittest.TestCase):
    def make(self, **fake):
        self.fake = FakeRobotiq(verbose=False, **fake)
        gripper = Gripper("", speed=200, force=100, poll_hz=200, fake=self.fake)
        gripper.start()
        self.addCleanup(gripper.close)
        return gripper

    def test_press_toggles_and_switches_direction(self):
        g = self.make()
        self.assertEqual(g.press("close"), "closing")  # mode is the commanded action, immediately
        self.assertTrue(wait_for(lambda: (g.snapshot()["pos"] or 0) > 20))
        self.assertEqual(g.press("close"), "stopped")
        self.assertTrue(wait_for(lambda: self.fake.sent[-1] == "SET GTO 0"))
        self.assertEqual(self.fake.sent[0], "SET POS 255 SPE 200 FOR 100 GTO 1")
        self.assertEqual(g.press("close"), "closing")
        self.assertEqual(g.press("open"), "opening")  # other button reverses directly
        self.assertTrue(wait_for(lambda: self.fake.sent[-1].startswith("SET POS 0 ")))

    def test_auto_stop_on_contact_and_at_travel_end(self):
        g = self.make(contact_at=120)
        g.press("close")
        self.assertTrue(wait_for(lambda: g.snapshot()["mode"] == "stopped"))
        snap = g.snapshot()
        self.assertEqual((snap["pos"], snap["obj"]), (120, 2))  # gripped an object
        g.press("open")
        self.assertTrue(wait_for(lambda: g.snapshot()["mode"] == "stopped"))
        snap = g.snapshot()
        self.assertEqual((snap["pos"], snap["obj"]), (0, 3))  # fully open
        self.assertIsNone(g.error)

    def test_open_never_toggles_to_stop(self):
        g = self.make()
        self.assertEqual(g.open(), "opening")
        self.assertEqual(g.open(), "opening")  # a second open keeps opening; the open button would stop it
        g.press("close")
        self.assertEqual(g.open(), "opening")  # reverses a close
        self.assertTrue(wait_for(lambda: g.snapshot()["mode"] == "stopped"))
        self.assertEqual((g.snapshot()["pos"], g.snapshot()["obj"]), (0, 3))  # fully open
        self.assertNotIn("SET GTO 0", self.fake.sent)

    def test_inactive_gripper_refused(self):
        fake = FakeRobotiq(verbose=False)
        fake.io = lambda command: "STA 0" if command == "GET STA" else "ack"
        with self.assertRaisesRegex(RuntimeError, "not activated"):
            Gripper("", speed=64, force=100, fake=fake).start()


if __name__ == "__main__":
    unittest.main()
