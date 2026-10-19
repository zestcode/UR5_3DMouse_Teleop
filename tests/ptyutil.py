"""Test helper: run a command with a pseudo-terminal as its controlling terminal, type keys into it (key) and read
what the processes write to it (out / wait_for): operator questions, status lines, the keyboard node's input."""
import fcntl
import os
import signal
import subprocess
import termios
import threading
import time


class PtyProcess:
    def __init__(self, cmd, env, cwd=None, stdout=None, stderr=None):
        """stdout / stderr: default the terminal (as in a shell); e.g. a log file or subprocess.PIPE instead."""
        self.master, slave = os.openpty()
        self.out = b""
        self.proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=slave,
                                     stdout=slave if stdout is None else stdout,
                                     stderr=slave if stderr is None else stderr, start_new_session=True,
                                     preexec_fn=lambda: fcntl.ioctl(0, termios.TIOCSCTTY, 0))
        os.close(slave)
        threading.Thread(target=self.drain, daemon=True).start()

    def drain(self):
        while True:
            try:
                data = os.read(self.master, 4096)
            except OSError:  # EIO: no process has the terminal open any more
                return
            if not data:
                return
            self.out += data

    def key(self, data):
        os.write(self.master, data)

    def wait_for(self, text, start=0, timeout=30):
        """-> position after the first `text` in the terminal output after `start`."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            i = self.out.find(text.encode(), start)
            if i >= 0:
                return i + len(text.encode())
            time.sleep(.05)
        raise AssertionError(f"{text!r} not on the terminal within {timeout} s; last output: {self.out[-1500:]!r}")

    def stop(self):
        """SIGINT to the process group (like Ctrl-C to `dora run`), wait, close the terminal."""
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        os.close(self.master)
