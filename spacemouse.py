"""spnav backend, based on the event/axis conventions of the supplied reference.

The last motion event is held until the next one, as DP / UMI do: spacenavd sends nothing while
the cap is held still, so "no new event" means "same deflection", not "released" (releasing
produces events back inside the deadzone). A dead daemon is detected on its socket instead.
"""
import os
import socket
import time
import numpy as np
from dora import Node
from common import map_motion, number, send_json, vector


def daemon_alive(sock):
    """Peek (never consume) without blocking: EOF or a socket error means spacenavd is gone."""
    try:
        return sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) != b""
    except BlockingIOError:
        return True  # connected, nothing pending
    except OSError:
        return False


def main():
    mock = os.getenv("MOCK_MOUSE", "0") == "1"
    mock_axes = os.getenv("MOCK_AXES", "") if mock else ""  # tests: the mock cap held at this deflection, enabled
    maximum = number("MOUSE_MAX", 300)
    deadzone = float(os.getenv("DEADZONE", "0.2"))
    # Button press (edge) -> event; no enable button: pushing the cap moves the arm.
    presses = {int(os.getenv("GRIPPER_OPEN_BUTTON", "0")): ("gripper", "open"),
               int(os.getenv("GRIPPER_CLOSE_BUTTON", "1")): ("gripper", "close")}
    record = int(os.getenv("RECORD_BUTTON", "-1"))  # -1: no record button
    if len(presses) != 2 or record in presses:
        raise ValueError("Gripper open/close and record buttons must all differ")
    if record >= 0:
        presses[record] = ("record", None)
    map_motion([0] * 6, maximum, deadzone)
    sock = None
    if not mock:
        import spnav
        spnav.spnav_open()
        sock = socket.socket(fileno=os.dup(spnav.spnav_fd()))  # own handle on the daemon socket, for peeking
    node = Node()
    axes, buttons = vector("MOCK_AXES", mock_axes, 6) if mock_axes else np.zeros(6), {}
    try:
        for event in node:
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT":
                continue
            if not mock:
                # Bound draining so a bad device cannot prevent command publication.
                received = False
                for _ in range(256):
                    sample = spnav.spnav_poll_event()
                    if sample is None:
                        break
                    received = True
                    if isinstance(sample, spnav.SpnavMotionEvent):
                        axes = map_motion(list(sample.translation) + list(sample.rotation), maximum, deadzone)
                    elif isinstance(sample, spnav.SpnavButtonEvent):
                        pressed = bool(sample.press)
                        if pressed and not buttons.get(sample.bnum, False) and sample.bnum in presses:
                            output, button = presses[sample.bnum]
                            send_json(node, output, {"t_ns": time.monotonic_ns(), "button": button})
                        buttons[sample.bnum] = pressed
                if not received and not daemon_alive(sock):
                    raise RuntimeError("spacenavd connection lost")
            # Every tick: t_ns lets ur5 detect a dead node; the axes are the last deflection.
            send_json(node, "command", {"t_ns": time.monotonic_ns(), "axes": axes.tolist(),
                                        "enabled": not mock or bool(mock_axes)})
    finally:
        if not mock:
            sock.close()
            spnav.spnav_close()


if __name__ == "__main__":
    main()
