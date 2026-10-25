"""Read-only checks run by teleop.sh before the dataflow: robot RUNNING, Robotiq activated (both skipped
with DRY_RUN != 0), cameras connected and not held by another process; a camera on USB 2 (usually a bad cable) is
accepted only if a 2 s test stream in the recording mode (COLOR_W x COLOR_H @ FPS) delivers >= 90 % of FPS.
One line per check; exit status 1 on the first blocking problem."""
import os
from pathlib import Path
import socket
import sys
import time

ROBOT_IP = os.getenv("ROBOT_IP", "")  # from teleop.local.env
CAMERAS = os.getenv("CAMERAS", "")    # role:serial,... from teleop.local.env
SIZE = (int(os.getenv("COLOR_W", "640")), int(os.getenv("COLOR_H", "480")))  # recorder.py defaults
FPS = int(os.getenv("FPS", "30"))


def ok(text):
    print(f"  ✓ {text}", flush=True)


def fail(text):
    print(f"  ✗ {text}", flush=True)
    sys.exit(1)


def video_holders():
    """{pid: command line} of processes with a /dev/video* node open (own processes only)."""
    out = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if any(os.readlink(fd).startswith("/dev/video") for fd in (p / "fd").iterdir()):
                out[int(p.name)] = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")[:120]
        except OSError:
            continue
    return out


def stream_fps(rs, serial, seconds=2.0):
    """Color frames per second the camera delivers in the recording mode; 0 if it does not stream."""
    pipe, config = rs.pipeline(), rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, SIZE[0], SIZE[1], rs.format.bgr8, FPS)
    try:
        pipe.start(config)
    except RuntimeError:
        return 0.0
    try:
        pipe.wait_for_frames(5000)  # first frame: auto exposure settles
        n, end = 0, time.monotonic() + seconds
        while time.monotonic() < end:
            pipe.wait_for_frames(2000)
            n += 1
        return n / seconds
    except RuntimeError:  # Frame didn't arrive within ...
        return 0.0
    finally:
        pipe.stop()


def configured(name, value):
    if not value or "<" in value:
        fail(f"{name} not configured: set it in teleop.local.env (copy teleop.local.env.example)")
    return value


def main():
    mock = os.getenv("MOCK_CAMERA", "0") == "1"
    cameras = {} if mock else dict(item.split(":") for item in configured("CAMERAS", CAMERAS).split(","))
    if os.getenv("DRY_RUN", "0") != "0":
        ok("DRY_RUN: not connecting to the robot (robot and gripper checks skipped)")
    else:
        configured("ROBOT_IP", ROBOT_IP)
        from rtde_receive import RTDEReceiveInterface
        try:
            receive = RTDEReceiveInterface(ROBOT_IP)
        except Exception as exc:
            fail(f"Cannot connect to the robot {ROBOT_IP}: {exc}")
        mode, safety = receive.getRobotMode(), receive.getSafetyMode()
        receive.disconnect()
        if mode != 7 or safety not in (1, 2):
            fail(f"Robot not ready (robot_mode {mode}, safety_mode {safety}): power on and release the brakes on the teach pendant, or clear the protective stop")
        ok("Robot RUNNING, safety mode normal")
        try:
            with socket.create_connection((ROBOT_IP, 63352), timeout=2) as sock:
                sock.sendall(b"GET STA\n")
                status = sock.recv(64).decode().split()
        except OSError as exc:
            fail(f"Cannot connect to the Robotiq ({ROBOT_IP}:63352): {exc}")
        if status != ["STA", "3"]:
            fail(f"Gripper not activated ({' '.join(status)}): teach pendant Robotiq toolbar -> Activate")
        ok("Robotiq gripper activated")
    if mock:
        ok("MOCK_CAMERA: cameras not checked")
        return
    import pyrealsense2 as rs
    found = {d.get_info(rs.camera_info.serial_number): d for d in rs.context().query_devices()}
    usb2 = []
    for role, serial in cameras.items():
        if serial not in found:
            fail(f"{role} camera {serial} not connected")
        usb = found[serial].get_info(rs.camera_info.usb_type_descriptor)
        if not usb.startswith("3"):
            usb2.append((role, serial, usb))
    ok(f"{len(cameras)} D455 connected" + ("" if usb2 else " (USB 3)"))
    holders = video_holders()
    if holders:
        pid, cmd = next(iter(holders.items()))
        fail(f"Cameras in use by another program: PID {pid} {cmd}")
    ok("Cameras free")
    for role, serial, usb in usb2:
        fps = stream_fps(rs, serial)
        mode = f"{SIZE[0]}x{SIZE[1]} @ {FPS} fps"
        if fps < 0.9 * FPS:
            fail(f"{role} camera {serial} is on USB {usb} and streams {fps:.0f} of {mode}: reseat or replace its cable")
        ok(f"{role} camera {serial} streams {fps:.0f} of {mode} on USB {usb} (not 3: check its cable)")


if __name__ == "__main__":
    main()
