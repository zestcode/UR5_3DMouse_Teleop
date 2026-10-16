"""Keyboard node: reads Esc from the terminal running `dora run` (works in a VS Code
Remote-SSH terminal; no X display, no pynput), polled on its timer tick.

Esc -> `esc` output (ur5 returns home, then stops the dataflow), then the node exits.
RECORD_KEY=1: Space -> `record` output (start / stop an episode; declare the output in the graph).
While running, the node holds the terminal foreground in cbreak mode (typing is not
echoed), so Ctrl-C reaches this node; it forwards it to `dora run` = stop without homing.
The terminal is restored on exit; if it is ever left in a bad state, run `stty sane`.
"""
import os
import select
import signal
import termios
import time
import tty
import pyarrow as pa
from dora import Node

ESC = b"\x1b"


def lone_esc(pending, data, esc_at, now, wait=0.1):
    """-> (pending, esc_at, pressed). A lone ESC byte with nothing after it for `wait` s is
    the Esc key; ESC followed by more bytes is a key sequence (arrows, F-keys, Alt+key)."""
    pending += data
    last = pending.rfind(ESC)
    pending = pending[last:] if last >= 0 else b""
    if pending != ESC:
        return b"", None, False
    if esc_at is None:
        return pending, now, False
    return (b"", None, True) if now - esc_at >= wait else (pending, esc_at, False)


class Terminal:
    """The controlling terminal in cbreak mode, with this node's process group in front."""

    def __init__(self):
        self.fd = os.open("/dev/tty", os.O_RDWR)
        self.attrs = termios.tcgetattr(self.fd)
        self.foreground = os.tcgetpgrp(self.fd)
        signal.signal(signal.SIGTTOU, signal.SIG_IGN)  # allow tcsetpgrp from a background group
        os.tcsetpgrp(self.fd, os.getpgrp())
        tty.setcbreak(self.fd)

    def read(self):
        return os.read(self.fd, 64) if select.select([self.fd], [], [], 0)[0] else b""

    def restore(self):
        for step in (lambda: termios.tcsetattr(self.fd, termios.TCSADRAIN, self.attrs),
                     lambda: os.tcsetpgrp(self.fd, self.foreground)):
            try:
                step()
            except OSError:  # e.g. the original foreground group has already exited
                pass
        os.close(self.fd)


def main():
    record_key = os.getenv("RECORD_KEY", "0") == "1"
    home_key = os.getenv("HOME_KEY", "0") == "1"  # h -> `home` output
    node = Node()
    try:
        term = Terminal()
    except OSError as exc:  # no controlling terminal (started detached / without a tty)
        term = None
        print(f"[Keyboard] disabled ({exc}); use Ctrl-C to stop", flush=True)
    else:
        parent = os.getppid()  # `dora run`

        def forward(*_):
            os.kill(parent, signal.SIGINT)  # Ctrl-C: dora stops every node, no homing
        signal.signal(signal.SIGINT, forward)
        signal.signal(signal.SIGQUIT, forward)
        signal.signal(signal.SIGTSTP, signal.SIG_IGN)
        print("[Keyboard] Esc (in this terminal) = return home and stop all nodes; "
              "Ctrl-C = stop without homing" + ("; Space = start/stop an episode" if record_key else "") + ("; h = home" if home_key else ""), flush=True)
    pending, esc_at = b"", None
    try:
        for event in node:
            if event["type"] == "STOP":
                break
            if term is None or event["type"] != "INPUT":
                continue
            data = term.read()
            for _ in range(data.count(b" ") if record_key else 0):
                node.send_output("record", pa.array([0]))
            if home_key and (b"h" in data or b"H" in data):
                node.send_output("home", pa.array([0]))
            pending, esc_at, pressed = lone_esc(pending, data, esc_at, time.monotonic())
            if pressed:
                print("[Keyboard] Esc: returning home, then stopping all nodes", flush=True)
                node.send_output("esc", pa.array([0]))
                break
    finally:
        if term is not None:
            term.restore()


if __name__ == "__main__":
    main()
