import unittest
from keyboard import delete_keys, enter_keys, lone_esc


def feed(steps):
    """steps: (data, t) per tick -> tick index where Esc was detected, or None."""
    pending, esc_at = b"", None
    for i, (data, now) in enumerate(steps):
        pending, esc_at, pressed = lone_esc(pending, data, esc_at, now)
        if pressed:
            return i
    return None


class EscTests(unittest.TestCase):
    def test_lone_esc_after_wait(self):
        self.assertEqual(feed([(b"\x1b", 0.0), (b"", 0.05), (b"", 0.11)]), 2)

    def test_arrow_keys_are_not_esc(self):
        self.assertIsNone(feed([(b"\x1b[A", 0.0), (b"", 0.2)]))  # one read
        self.assertIsNone(feed([(b"\x1b", 0.0), (b"[B", 0.01), (b"", 0.3)]))  # split read
        self.assertIsNone(feed([(b"\x1bOP", 0.0), (b"", 0.3)]))  # F1

    def test_alt_key_is_not_esc(self):
        self.assertIsNone(feed([(b"\x1bq", 0.0), (b"", 0.3)]))

    def test_plain_keys_ignored_and_esc_after_them(self):
        self.assertIsNone(feed([(b"abc", 0.0), (b"", 0.3)]))
        self.assertEqual(feed([(b"q\x1b", 0.0), (b"", 0.2)]), 1)


class DeleteKeyTests(unittest.TestCase):
    def test_d_then_y_in_one_or_two_reads(self):
        self.assertEqual(delete_keys(b"dy", False), (["ask", "yes"], False))
        self.assertEqual(delete_keys(b"d", False), (["ask"], True))
        self.assertEqual(delete_keys(b"Y", True), (["yes"], False))

    def test_any_other_key_answers_no_once(self):
        self.assertEqual(delete_keys(b" ", True), (["no"], False))
        self.assertEqual(delete_keys(b"dny", False), (["ask", "no"], False))  # y after the answer: nothing

    def test_y_without_d_and_plain_keys_send_nothing(self):
        self.assertEqual(delete_keys(b"y h", False), ([], False))


class EnterKeyTests(unittest.TestCase):
    def test_each_enter_counts_once_whatever_the_line_ending(self):
        self.assertEqual(enter_keys(b"\n"), 1)  # cbreak with ICRNL (the usual terminal)
        self.assertEqual(enter_keys(b"\r"), 1)  # a terminal without ICRNL
        self.assertEqual(enter_keys(b"\r\n"), 1)
        self.assertEqual(enter_keys(b"\n \nx\r"), 3)  # several in one read, between other keys

    def test_other_keys_are_not_enter(self):
        self.assertEqual(enter_keys(b""), 0)
        self.assertEqual(enter_keys(b" dyh\x1b[A"), 0)


if __name__ == "__main__":
    unittest.main()
