"""Robotiq gripper through the Robotiq URCap socket on the UR controller (port 63352).

Owned by the ur5 node so gripper action/feedback land in the same state record as
the arm. A press starts opening/closing at a fixed speed; pressing the same button
again stops (GTO 0). The firmware stops by itself on contact (OBJ 1 opening / 2
closing, keeping FOR force) or at the end of travel (OBJ 3). `mode` is the commanded
action and changes in the caller's thread; `target` the commanded position (0 open ..
255 closed; None once stopped with the button: it stays where it is; a firmware stop
keeps it: on an object the gripper still squeezes). A background thread does the socket
I/O so the RTDE loop never blocks on it.
"""
import queue
import socket
import threading
import time

TARGET = {"open": 0, "close": 255}
MOVING = {"open": "opening", "close": "closing"}


class FakeRobotiq:
    """Dry-run stand-in for the URCap socket: prints SETs, moves at a speed-proportional rate."""

    def __init__(self, contact_at=None, verbose=True):
        self.contact_at, self.verbose = contact_at, verbose
        self.pos, self.target, self.gto, self.speed = 0.0, 0, 0, 0
        self.t = time.monotonic()
        self.sent = []

    def _advance(self):
        now = time.monotonic()
        if self.gto:
            step = (40 + self.speed) * (now - self.t)
            if self.target > self.pos:
                limit = self.target if self.contact_at is None else min(self.target, self.contact_at)
                self.pos = min(limit, self.pos + step)
            else:
                self.pos = max(self.target, self.pos - step)
        self.t = now

    def io(self, command):
        self._advance()
        if command.startswith("SET"):
            self.sent.append(command)
            if self.verbose:
                print(f"DRY gripper -> {command}", flush=True)
            words = command.split()[1:]
            for name, value in zip(words[::2], words[1::2]):
                if name == "POS":
                    self.target = int(value)
                elif name == "SPE":
                    self.speed = int(value)
                elif name == "GTO":
                    self.gto = int(value)
            return "ack"
        name = command.split()[1]
        blocked = self.contact_at is not None and self.target > self.contact_at and self.pos >= self.contact_at
        moving = self.gto and round(self.pos) != self.target and not blocked
        value = {"STA": 3, "POS": round(self.pos), "PRE": self.target,
                 "OBJ": 0 if moving else 2 if blocked else 3}.get(name, 0)
        return f"{name} {value}"


class Gripper:
    def __init__(self, host, speed, force, port=63352, poll_hz=50, fake=None):
        if not (1 <= speed <= 255 and 1 <= force <= 255):
            raise ValueError("GRIPPER_SPEED and GRIPPER_FORCE must be in 1..255")
        self.host, self.port, self.speed, self.force = host, port, int(speed), int(force)
        self.period = 1 / poll_hz
        self.fake = fake
        self.sock = None
        self.lock = threading.Lock()
        self.commands = queue.SimpleQueue()
        self.stop_event = threading.Event()
        self.thread = None
        self.mode, self.pos, self.obj, self.t_ns = "stopped", None, None, None
        self.target = None
        self.error = None
        # Auto-stop is trusted only once the latest press was sent, echoed (PRE) and 0.1 s old.
        self.seq = self.sent_seq = 0
        self.pending = None
        self.sent_at = 0.0

    def io(self, command):
        if self.fake is not None:
            return self.fake.io(command)
        self.sock.sendall((command + "\n").encode())
        return self.sock.recv(1024).decode().strip()

    def get(self, name):
        reply = self.io(f"GET {name}").split()
        if len(reply) != 2 or reply[0] != name:
            raise RuntimeError(f"Unexpected gripper reply to GET {name}: {reply}")
        return int(reply[1])

    def set(self, command):
        if self.io(command) != "ack":
            raise RuntimeError(f"Gripper did not ack '{command}'")

    def start(self):
        if self.fake is None:
            self.sock = socket.create_connection((self.host, self.port), timeout=2)
        if self.get("STA") != 3:
            raise RuntimeError("Robotiq gripper is not activated; activate it on the teach pendant first")
        self.poll()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def press(self, button, toggle=True):
        """Toggle on a button press; returns the new commanded mode."""
        with self.lock:
            self.seq += 1
            if toggle and self.mode == MOVING[button]:
                self.mode, command, target = "stopped", "SET GTO 0", None
            else:
                target = TARGET[button]
                self.mode = MOVING[button]
                command = f"SET POS {target} SPE {self.speed} FOR {self.force} GTO 1"
            self.target = target
            self.commands.put((self.seq, command, target))
            return self.mode

    def open(self):
        """Open fully, also when already opening (episode start state; a button press there would stop it)."""
        return self.press("open", toggle=False)

    def poll(self):
        with self.lock:
            pending = self.pending if self.sent_seq == self.seq else None
        acknowledged = pending is not None and self.get("PRE") == pending
        pos, obj, t_ns = self.get("POS"), self.get("OBJ"), time.monotonic_ns()
        with self.lock:
            if acknowledged and self.pending == pending:
                self.pending = None
            if (self.mode != "stopped" and self.sent_seq == self.seq and self.pending is None
                    and time.monotonic() - self.sent_at >= 0.1 and obj != 0):
                self.mode = "stopped"  # firmware stopped: contact (1/2) or end of travel (3)
            self.pos, self.obj, self.t_ns = pos, obj, t_ns

    def run(self):
        try:
            while not self.stop_event.is_set():
                start = time.monotonic()
                while True:
                    try:
                        seq, command, target = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    self.set(command)
                    with self.lock:
                        self.sent_seq, self.pending, self.sent_at = seq, target, time.monotonic()
                self.poll()
                time.sleep(max(0, self.period - (time.monotonic() - start)))
        except Exception as exc:  # surfaced to the control loop, which stops the arm
            self.error = repr(exc)

    def snapshot(self):
        with self.lock:
            return {"mode": self.mode, "pos": self.pos, "obj": self.obj, "t_ns": self.t_ns, "target": self.target}

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        try:
            if self.error is None and self.mode != "stopped":
                self.set("SET GTO 0")  # stop an unfinished open/close; a held grip is left as is
        finally:
            if self.sock is not None:
                self.sock.close()
