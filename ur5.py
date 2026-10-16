"""One owner of RTDEControl; local periodic loop, independently of Dora timers.

Option B (as DP / UMI on UR): the mouse twist is integrated into a TARGET pose (never the
measured pose), clamped to the workspace and leashed to the robot, and streamed with servoL
at CONTROL_HZ; the UR controller smooths it (lookahead). No speedL / blocking speedStop in the loop.
Real mode is paced by the controller's own RTDE packets, not a host timer: one servoL per controller
cycle at a fixed phase, so the host and controller clocks never slip against each other.
"""
import os
import signal
import time
import numpy as np
from dora import Node
from common import (clamp_box, command_live, flange_from_tcp, follow_leash, integrate_pose, limited_twist,
                    matrix_pose, number, pose9d, pose_error, pose_matrix, read_json, send_json, smooth_twist,
                    vector, workspace_bounds, within_workspace)
from gripper import FakeRobotiq, Gripper


def checked(result, action):
    if not result:
        raise RuntimeError(f"ur_rtde {action} failed")


def is_timeout(event):
    # Dora 0.3.8 represents receive timeout as an ERROR event.
    return event is None or (event["type"] == "ERROR" and ("timeout" in str(event).lower()
                                                            or "timed out" in str(event).lower()))


def wait_for_packet(receive, last, timeout_s, poll_s=0.0002):
    """Block until RTDE delivers a controller packet newer than `last`; return its timestamp.
    The controller's 125 Hz output is the loop clock (ur_rtde initPeriod/waitPeriod time each
    cycle from its own start, so loop overhead accumulates and the phase drifts, ~0.1% measured)."""
    deadline = time.monotonic() + timeout_s
    while (stamp := receive.getTimestamp()) == last:
        if time.monotonic() > deadline:
            raise RuntimeError("RTDE feedback stale")
        time.sleep(poll_s)
    return stamp


def next_tick(deadline, period, now):
    """Dry run pacing on absolute deadlines (no drift from loop overhead); more than a period
    behind, resync instead of running a burst of cycles."""
    deadline += period
    return now if deadline < now - period else deadline


def stop_requested(node, seconds):
    """Wait up to `seconds` for Dora STOP. Ctrl-C reaches nodes only as STOP (each node has
    its own process group), so blocking phases must poll for it. Other events are dropped,
    including INPUT_CLOSED (the keyboard node exits right after sending Esc)."""
    end = time.monotonic() + seconds
    while (remaining := end - time.monotonic()) > 0:
        event = node.next(timeout=min(remaining, 0.05))
        if is_timeout(event):
            continue
        if event["type"] == "ERROR":
            raise RuntimeError(str(event))
        if event["type"] == "STOP":
            return True
    return False


def move_home(node, control, receive, home, speed, acceleration, delay, watchdog=False):
    """Asynchronous moveJ so STOP can abort it with stopJ. `watchdog`: kick the RTDE
    watchdog while waiting (needed once setWatchdog is active, i.e. on Esc)."""
    q0 = np.array(receive.getActualQ())
    if not control.isJointsWithinSafetyLimits(home.tolist()):
        raise RuntimeError("HOME_Q_DEG is outside the joint safety limits")
    print(f"Homing in {delay:g} s (Ctrl-C aborts): q_deg {np.degrees(q0).round(2).tolist()} "
          f"-> {np.degrees(home).round(2).tolist()}", flush=True)
    if stop_requested(node, delay):
        return False
    deadline = time.monotonic() + 10 + 2 * np.abs(home - q0).max() / speed
    checked(control.moveJ(home.tolist(), speed, acceleration, True), "moveJ home")
    try:
        while not (np.abs(np.array(receive.getActualQ()) - home).max() < 0.005
                   and np.abs(receive.getActualQd()).max() < 0.01):
            if receive.getRobotMode() != 7 or receive.getSafetyMode() not in (1, 2):
                raise RuntimeError("Robot mode/safety mode changed while homing")
            if time.monotonic() > deadline:
                raise RuntimeError("Homing did not reach HOME_Q_DEG in time")
            if watchdog:
                checked(control.kickWatchdog(), "kickWatchdog")
            if stop_requested(node, 0.02):
                control.stopJ(2.0)
                return False
    except BaseException:
        control.stopJ(2.0)
        raise
    print(f"Home reached: tcp {np.round(receive.getActualTCPPose(), 4).tolist()}", flush=True)
    return True


def main():
    real = os.getenv("DRY_RUN", "1") == "0"
    hz = number("CONTROL_HZ", 125)
    if hz not in (125, 500):
        raise ValueError("CONTROL_HZ must be 125 (CB3/e) or 500 (e-series)")
    dt = 1 / hz
    linear, angular = number("MAX_LINEAR_M_S", .03), number("MAX_ANGULAR_RAD_S", .15)
    stop_acceleration = number("STOP_ACCELERATION", 1.)  # servoStop on Esc / exit
    lookahead, gain = number("SERVO_LOOKAHEAD_S", .1), number("SERVO_GAIN", 300)  # DP/UMI: 0.1 s, 300
    if not (.03 <= lookahead <= .2 and 100 <= gain <= 2000):
        raise ValueError("servoL needs SERVO_LOOKAHEAD_S in [0.03, 0.2] and SERVO_GAIN in [100, 2000]")
    leash_m, leash_rad = number("LEASH_M", .03), number("LEASH_RAD", .2)
    tau = number("TWIST_FILTER_S", .15)  # low-pass on the mouse twist; release ramps down, not speedStop(1.0)
    timeout = number("COMMAND_TIMEOUT_S", .15)
    recorder_timeout = number("RECORDER_TIMEOUT_S", 1)
    margin = number("BOUND_MARGIN_M", .02)
    watchdog = number("WATCHDOG_HZ", 5)
    if watchdog >= hz:
        raise ValueError("WATCHDOG_HZ must be below CONTROL_HZ")
    lo, hi = workspace_bounds(False)  # real mode recomputes once the start TCP is known
    use_gripper = os.getenv("GRIPPER", "0") == "1"
    grip_speed, grip_force = number("GRIPPER_SPEED", 64), number("GRIPPER_FORCE", 100)
    home_deg = os.getenv("HOME_Q_DEG", "")
    home = np.radians(vector("HOME_Q_DEG", home_deg, 6)) if home_deg else None
    home_speed, home_acceleration = number("HOME_SPEED", .2), number("HOME_ACCELERATION", .3)
    home_delay = number("HOME_DELAY_S", 3)
    node = Node()
    control = receive = gripper = None
    command = None
    heartbeat = last_print = 0
    servoing = False
    twist = np.zeros(6)  # smoothed twist integrated into the target
    pose = np.array([0., 0., .5, 0., 0., 0.])
    tcp_offset = [0.] * 6  # T_flange_tcp from the pendant's TCP setting; dry run: flange = TCP
    robot_time = last_robot_time = None
    try:
        if use_gripper:  # before RTDE control: refuse to start with an inactive gripper
            gripper = Gripper(os.getenv("ROBOT_IP", ""), grip_speed, grip_force,
                              fake=None if real else FakeRobotiq())
            gripper.start()
        if real:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface
            ip = os.getenv("ROBOT_IP", "")
            if not ip:
                raise ValueError("Set ROBOT_IP")
            receive = RTDEReceiveInterface(ip, hz)
            if receive.getRobotMode() != 7 or receive.getSafetyMode() not in (1, 2):
                raise RuntimeError("Robot must be running, in normal/reduced safety mode")
            control = RTDEControlInterface(ip, hz)
            tcp_offset = [float(v) for v in control.getTCPOffset()]  # read once; flange pose is derived per cycle
            if len(tcp_offset) != 6 or not np.isfinite(tcp_offset).all():
                raise RuntimeError(f"Invalid TCP offset from controller: {tcp_offset}")
            print(f"TCP offset (flange frame, m / rotvec): {np.round(tcp_offset, 5).tolist()}", flush=True)
            if home is not None and not move_home(node, control, receive, home,
                                                  home_speed, home_acceleration, home_delay):
                return
            # Box around the TCP at home (or at start if no home is configured).
            pose = np.array(receive.getActualTCPPose())
            lo, hi = workspace_bounds(True, pose[:3])
            print(f"TCP workspace (base frame, m): min {np.round(lo, 3).tolist()} max {np.round(hi, 3).tolist()}", flush=True)
            checked(control.setWatchdog(watchdog), "setWatchdog")  # after homing: moveJ does not kick it
            last_robot_time = receive.getTimestamp()
        elif home is not None:
            print(f"DRY homing skipped: would moveJ to q_deg {np.degrees(home).round(2).tolist()}", flush=True)
        T_target = pose_matrix(pose)  # integrate on the target, never on the measured pose (DP)
        print(f"servoL: {hz:g} Hz {'paced by controller packets' if real else 'local clock'}, lookahead {lookahead:g} s, "
              f"gain {gain:g}; leash {leash_m:g} m / {leash_rad:g} rad", flush=True)
        esc = go_home = False
        tick = time.monotonic()
        while True:
            if real:
                # Wait for the controller's next packet; integrate over controller time (2 x dt after a missed one).
                robot_time = wait_for_packet(receive, last_robot_time, timeout)
                step = min(max(robot_time - last_robot_time, 0.), 3 * dt)
                last_robot_time = robot_time
            else:
                tick = next_tick(tick, dt, time.monotonic())
                time.sleep(max(0., tick - time.monotonic()))
                step = dt
            gripper_events = []
            # Small bounded polling budget; no Dora timer controls the RTDE period.
            for _ in range(4):
                event = node.next(timeout=0.00001)
                if is_timeout(event):
                    break
                if event["type"] == "ERROR":
                    raise RuntimeError(str(event))
                if event["type"] == "STOP" or (event["type"] == "INPUT_CLOSED" and event["id"] != "esc"):
                    return
                if event["type"] == "INPUT":
                    if event["id"] == "command":
                        command = read_json(event)
                    elif event["id"] == "heartbeat":
                        heartbeat = read_json(event)["t_ns"]
                    elif event["id"] == "gripper" and gripper is not None:
                        button = read_json(event)["button"]
                        gripper.press(button)  # commanded mode changes in this cycle
                        gripper_events.append(button)
                    elif event["id"] == "home":
                        go_home = True
                    elif event["id"] == "esc":
                        esc = True
                        break
            if esc:
                break
            if go_home:  # h key: moveJ home, then teleop again
                go_home = False
                if real and home is not None:
                    if servoing:
                        checked(control.servoStop(stop_acceleration), "servoStop")
                        servoing = False
                    print("h: returning home", flush=True)
                    if not move_home(node, control, receive, home, home_speed, home_acceleration, 0, watchdog=True):
                        return
                    pose = np.array(receive.getActualTCPPose())
                    T_target, twist = pose_matrix(pose), np.zeros(6)
                    last_robot_time = receive.getTimestamp()
                    continue
                if home is not None:
                    print(f"DRY h: would moveJ home to q_deg {np.degrees(home).round(2).tolist()}", flush=True)
            now = time.monotonic_ns()
            target = limited_twist(command, now, timeout, linear, angular)
            recorder_ok = 0 <= now - heartbeat <= recorder_timeout * 1e9
            # Disabled, stale or no recorder: freeze the target at once. Deadzone release: ramp down via the filter.
            hard_stop = not recorder_ok or not command_live(command, now, timeout)
            if hard_stop:
                target[:] = 0
                twist = np.zeros(6)
            else:
                twist = smooth_twist(twist, target, step, tau)
                if (not np.any(target) and np.linalg.norm(twist[:3]) < 5e-4
                        and np.linalg.norm(twist[3:]) < 5e-3):
                    twist = np.zeros(6)  # tail of the ramp: below 0.5 mm/s and 0.005 rad/s
            if gripper is not None and gripper.error:
                raise RuntimeError(f"Gripper failed: {gripper.error}")
            grip = {**gripper.snapshot(), "events": gripper_events} if gripper is not None else None
            if real:
                if not receive.isConnected() or not control.isConnected():
                    raise RuntimeError("RTDE disconnected")
                if receive.getRobotMode() != 7 or receive.getSafetyMode() not in (1, 2):
                    raise RuntimeError("Robot mode/safety mode changed; manual restart required")
                pose = np.array(receive.getActualTCPPose())
                joints = receive.getActualQ()
                actual_speed = receive.getActualTCPSpeed()
            # Target: integrate, clamp to the workspace (slides along the boundary), leash to the robot.
            T_target = follow_leash(T_target, clamp_box(integrate_pose(T_target, twist, step), lo, hi, margin, T_target),
                                    pose_matrix(pose) if real else None, leash_m, leash_rad)
            if not real:
                # Dry run: an ideal follower, so the target logic (clamp, integration) runs offline too.
                pose, joints, actual_speed = matrix_pose(T_target), [0.] * 6, twist.tolist()
            if now - last_print >= 200_000_000:  # 5 Hz status line in the dora console
                last_print = now
                err_m, err_rad = pose_error(T_target, pose_matrix(pose))
                status = (f"mouse={np.round(command['axes'] if command else [0.] * 6, 2).tolist()} "
                          f"enabled={bool(command and command['enabled'])} "
                          f"recorder={'ok' if recorder_ok else 'STALE'} target={'FROZEN' if hard_stop else 'live'} "
                          f"twist={np.round(twist, 4).tolist()} "
                          f"target_err={err_m * 1e3:.1f}mm/{np.degrees(err_rad):.2f}deg")
                if grip is not None:
                    status += f" gripper={grip['mode']} pos={grip['pos']} obj={grip['obj']}"
                print(f"REAL {status} tcp_xyz={np.round(pose[:3], 4).tolist()}" if real else f"DRY {status}", flush=True)
            # Safety net: the target is clamped, so the TCP itself leaving the box means something is wrong.
            if not within_workspace(pose, np.zeros(6), lo, hi, 0.):
                raise RuntimeError(f"TCP {np.round(pose[:3], 4).tolist()} outside configured workspace min "
                                   f"{np.round(lo, 4).tolist()} max {np.round(hi, 4).tolist()} (target is clamped inside)")
            if real:
                checked(control.kickWatchdog(), "kickWatchdog")
                servoing = True  # set before the call so cleanup leaves servo mode even on failure
                # speed / acceleration 0.5 are dummies (unused by servoL), the values DP passes on its UR5.
                checked(control.servoL(matrix_pose(T_target).tolist(), .5, .5, dt, lookahead, gain), "servoL")
            send_json(node, "state", {"t_ns": now, "robot_timestamp_s": robot_time,
                      # Recorded poses: xyz + rot6d (first two rows of R). Velocities stay 6-D (m/s, rad/s).
                      "tcp_pose": pose9d(pose_matrix(pose)).tolist(),
                      "flange_pose": pose9d(flange_from_tcp(pose, tcp_offset)).tolist(),
                      "tcp_offset": pose9d(pose_matrix(tcp_offset)).tolist(), "q": joints, "tcp_speed": actual_speed,
                      "target_pose": pose9d(T_target).tolist(), "command_twist": twist.tolist(),
                      "target_twist": target.tolist(), "hard_stop": bool(hard_stop), "gripper": grip, "simulated": not real,
                      "robot_mode": receive.getRobotMode() if real else 7,
                      "safety_mode": receive.getSafetyMode() if real else 1})
        # Esc: leave servo mode, return home, then end the whole dataflow (Ctrl-C ends it without homing).
        if servoing:
            checked(control.servoStop(stop_acceleration), "servoStop")
            servoing = False
        if real and home is not None:
            print("Esc: returning home", flush=True)
            if not move_home(node, control, receive, home, home_speed, home_acceleration, 0, watchdog=True):
                return
        elif home is not None:
            print(f"DRY Esc: would moveJ home to q_deg {np.degrees(home).round(2).tolist()}", flush=True)
        print("Esc: stopping the dataflow", flush=True)
        os.kill(os.getppid(), signal.SIGINT)  # parent is `dora run`: stops every node, like Ctrl-C
    finally:
        try:
            if control is not None:
                try:
                    if servoing:
                        control.servoStop(stop_acceleration)
                finally:
                    try:
                        control.stopScript()
                    finally:
                        control.disconnect()
        finally:
            try:
                if receive is not None:
                    receive.disconnect()
            finally:
                if gripper is not None:
                    gripper.close()


if __name__ == "__main__":
    main()
