"""Pure helpers; translation in metres, angular velocity in radians/s, base frame."""
import json
import os
import numpy as np


def number(name, default):
    value = float(os.getenv(name, str(default)))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def map_motion(raw, maximum=300.0, deadzone=0.2, rotation_deadzone=None, rotation_curve=1.0,
               rotation_coupling=0.0):
    """Same spnav axis convention as zestcode/ur3-neu-dev: [-z, x, y].
    Rotation is shaped on its own (defaults: like translation): `rotation_deadzone`; output raised to
    `rotation_curve` (> 1: finer near the centre, full scale unchanged); and, since pushing the cap also
    tilts it, its deadzone edge moves out by `rotation_coupling` x the translation output norm."""
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (6,) or not np.isfinite(raw).all():
        raise ValueError("Expected six finite spnav axes")
    rotation_deadzone = deadzone if rotation_deadzone is None else rotation_deadzone
    if (maximum <= 0 or not 0 <= deadzone < 1 or not 0 <= rotation_deadzone < 1
            or not 0 < rotation_curve < np.inf or not 0 <= rotation_coupling < np.inf):
        raise ValueError("Invalid normalization, deadzone or rotation shaping")
    raw = np.clip(raw / maximum, -1, 1)
    # Rescale past the deadzone so output rises continuously from 0 at its edge (no step).
    move = np.sign(raw[:3]) * np.maximum(np.abs(raw[:3]) - deadzone, 0) / (1 - deadzone)
    # A moved edge keeps the same slope (no step either); while translating, full scale may not be reached.
    edge = rotation_deadzone + rotation_coupling * np.linalg.norm(move)
    turn = np.maximum(np.abs(raw[3:]) - edge, 0) / (1 - rotation_deadzone)
    return np.r_[move, np.sign(raw[3:]) * turn ** rotation_curve][[2, 0, 1, 5, 3, 4]] * [-1, 1, 1, -1, 1, 1]


def command_live(command, now_ns, timeout_s):
    """Enabled and fresh; otherwise the arm must stop at once (not ramp down)."""
    if command is None:
        return False
    age = (now_ns - command["t_ns"]) / 1e9
    return bool(command["enabled"]) and 0 <= age <= timeout_s


def smooth_twist(previous, target, dt, tau):
    """First-order low-pass, time constant `tau` s: no velocity steps on press or release."""
    previous, target = np.asarray(previous, dtype=float), np.asarray(target, dtype=float)
    return previous + (target - previous) * (1 - np.exp(-dt / tau))


def limited_twist(command, now_ns, timeout_s, linear, angular):
    zero = np.zeros(6)
    if not command_live(command, now_ns, timeout_s):
        return zero
    axes = np.asarray(command["axes"], dtype=float)
    if axes.shape != (6,) or not np.isfinite(axes).all():
        raise ValueError("Invalid mouse command")
    for part, limit in ((axes[:3], linear), (axes[3:], angular)):
        part *= limit / max(1.0, np.linalg.norm(part))
    return axes


def vector(name, text, size=3, null=None):
    """JSON array of `size` finite numbers. If `null` is given, JSON null entries become
    that value (workspace bounds: ±inf = no limit on that axis); otherwise null is invalid."""
    unlimited = []
    try:
        items = json.loads(text)
        unlimited = [item is None for item in items]
        value = np.array([0. if u else item for item, u in zip(items, unlimited)], dtype=float)
    except (ValueError, TypeError):
        value = None
    if (value is None or value.shape != (size,) or not np.isfinite(value).all()
            or (any(unlimited) and null is None)):
        raise ValueError(f"{name} must be a JSON array of {size} finite numbers"
                         + (" or null (no limit)" if null is not None else ""))
    if any(unlimited):
        value[unlimited] = null
    return value


def workspace_bounds(real, start_xyz=None):
    """Base-frame TCP box: WORKSPACE_MIN/MAX if set; otherwise, in real mode,
    the start TCP read over RTDE plus WORKSPACE_OFFSET_MIN/MAX. A null entry means no
    limit on that axis. WORKSPACE_Z_MIN, if set, replaces the bottom with an absolute
    base-frame z (e.g. a recorded floor)."""
    lo, hi = os.getenv("WORKSPACE_MIN", ""), os.getenv("WORKSPACE_MAX", "")
    if lo or hi:
        lo, hi = vector("WORKSPACE_MIN", lo, null=-np.inf), vector("WORKSPACE_MAX", hi, null=np.inf)
    elif real:
        start = np.asarray(start_xyz, dtype=float)
        lo = start + vector("WORKSPACE_OFFSET_MIN", os.getenv("WORKSPACE_OFFSET_MIN") or "[-0.15, -0.15, -0.05]",
                            null=-np.inf)
        hi = start + vector("WORKSPACE_OFFSET_MAX", os.getenv("WORKSPACE_OFFSET_MAX") or "[0.15, 0.15, 0.15]",
                            null=np.inf)
    else:
        return np.full(3, -2.0), np.full(3, 2.0)
    floor = os.getenv("WORKSPACE_Z_MIN", "")
    if floor:
        lo[2] = float(floor)
    if np.isnan([lo, hi]).any() or not (lo < hi).all():
        raise ValueError("Invalid workspace bounds")
    return lo, hi


def within_workspace(pose, twist, lo, hi, margin, horizon=0.25):
    xyz = np.asarray(pose[:3])
    predicted = xyz + np.asarray(twist[:3]) * horizon
    return bool(np.isfinite(pose).all() and (xyz >= lo + margin).all()
                and (xyz <= hi - margin).all() and (predicted >= lo + margin).all()
                and (predicted <= hi - margin).all())


def pose_matrix(pose):
    """UR pose [x, y, z, rx, ry, rz] (metres, rotation vector) -> 4x4 transform."""
    pose = np.asarray(pose, dtype=float)
    rotvec, angle = pose[3:], np.linalg.norm(pose[3:])
    K = np.array([[0, -rotvec[2], rotvec[1]], [rotvec[2], 0, -rotvec[0]], [-rotvec[1], rotvec[0], 0]])
    if angle < 1e-12:
        rot = np.eye(3) + K
    else:
        K = K / angle
        rot = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot, pose[:3]
    return T


def matrix_pose(T):
    """4x4 transform -> UR pose; via a quaternion, so angles near pi stay well conditioned."""
    m = np.asarray(T, dtype=float)[:3, :3]
    w = np.sqrt(max(0., 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0., 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = np.copysign(np.sqrt(max(0., 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = np.copysign(np.sqrt(max(0., 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    v = np.array([x, y, z])
    if w < 0.5:  # near pi the off-diagonal signs above are unreliable: rebuild v from the largest component
        i = int(np.argmax(np.abs(v)))
        v = np.empty(3)
        v[i] = np.sqrt(max(0., 1 + 2 * m[i, i] - np.trace(m))) / 2
        j, k = (i + 1) % 3, (i + 2) % 3
        v[j] = (m[j, i] + m[i, j]) / (4 * v[i])
        v[k] = (m[k, i] + m[i, k]) / (4 * v[i])
        w = (m[k, j] - m[j, k]) / (4 * v[i])
        if w < 0:
            v, w = -v, -w
    norm = np.linalg.norm(v)
    rotvec = np.zeros(3) if norm < 1e-12 else v / norm * 2 * np.arctan2(norm, w)
    return np.r_[np.asarray(T, dtype=float)[:3, 3], rotvec]


def flange_from_tcp(tcp_pose, tcp_offset):
    """4x4 T_base_flange = T_base_tcp @ inv(T_flange_tcp), with the TCP offset the controller is using."""
    return pose_matrix(tcp_pose) @ np.linalg.inv(pose_matrix(tcp_offset))


def integrate_pose(T, twist, dt):
    """Advance a 4x4 target by a base-frame twist [v, w] for dt: p += v dt, R <- exp(w dt) R.
    Same convention as DP/UMI demo_real_robot (drot * R): rotation about the TCP, axes in base."""
    twist = np.asarray(twist, dtype=float)
    out = np.array(T, dtype=float, copy=True)
    out[:3, :3] = pose_matrix(np.r_[0., 0., 0., twist[3:] * dt])[:3, :3] @ out[:3, :3]
    out[:3, 3] = out[:3, 3] + twist[:3] * dt
    return out


def clamp_box(T, lo, hi, margin, prev=None):
    """Keep the target translation inside [lo + margin, hi - margin]; +/-inf bounds pass through.
    With `prev` (the last target), an axis already outside that range may stay or move inward
    but not further out, so a start inside the margin is never snapped (no target jump)."""
    lo, hi = np.asarray(lo, dtype=float) + margin, np.asarray(hi, dtype=float) - margin
    if prev is not None:
        held = np.asarray(prev, dtype=float)[:3, 3]
        lo, hi = np.minimum(lo, held), np.maximum(hi, held)
    out = np.array(T, dtype=float, copy=True)
    out[:3, 3] = np.clip(out[:3, 3], lo, hi)
    return out


def pose_error(Ta, Tb):
    """(translation distance m, rotation angle rad) between two 4x4 poses."""
    Ta, Tb = np.asarray(Ta, dtype=float), np.asarray(Tb, dtype=float)
    cos = (np.trace(Ta[:3, :3].T @ Tb[:3, :3]) - 1) / 2
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), float(np.arccos(np.clip(cos, -1., 1.)))


def follow_leash(T_old, T_new, T_actual, max_pos, max_rot):
    """Keep the target within max_pos / max_rot of the measured pose: a step that would take it
    further outside is refused (robot not following), steps back toward the robot are allowed."""
    if T_actual is None:
        return T_new
    d_new, r_new = pose_error(T_new, T_actual)
    if d_new <= max_pos and r_new <= max_rot:
        return T_new
    d_old, r_old = pose_error(T_old, T_actual)
    return T_new if d_new <= max(d_old, max_pos) and r_new <= max(r_old, max_rot) else T_old


def mat_to_rot6d(rot):
    """rot6d = first two ROWS of R, as UMI pose_util.mat_to_rot6d / pytorch3d matrix_to_rotation_6d."""
    rot = np.asarray(rot, dtype=float)
    return rot[..., :2, :].reshape(rot.shape[:-2] + (6,))


def rot6d_to_mat(d6):
    """Inverse of mat_to_rot6d (Gram-Schmidt; rows b1, b2, b1 x b2), as UMI pose_util.rot6d_to_mat."""
    d6 = np.asarray(d6, dtype=float)
    b1 = d6[..., :3] / np.linalg.norm(d6[..., :3], axis=-1, keepdims=True)
    b2 = d6[..., 3:] - np.sum(b1 * d6[..., 3:], axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack((b1, b2, np.cross(b1, b2)), axis=-2)


def pose9d(T):
    """4x4 transform -> [x, y, z, rot6d] (9 values)."""
    T = np.asarray(T, dtype=float)
    return np.r_[T[:3, 3], mat_to_rot6d(T[:3, :3])]


def relative_rotvec(R0, R1):
    """Rotation vector of R0^T R1: R1 = R0 exp(rotvec), the shortest way from R0 to R1. Below 120 deg from the
    antisymmetric part (exactly 0 for R0 = R1; matrix_pose's square roots turn 1e-16 into 1e-8 there)."""
    R = np.asarray(R0, dtype=float).T @ np.asarray(R1, dtype=float)
    v = .5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])  # sin(angle) x axis
    s, c = np.linalg.norm(v), (np.trace(R) - 1) / 2
    if c > -.5:
        return v * (np.arctan2(s, c) / s) if s > 1e-15 else v
    T = np.eye(4)
    T[:3, :3] = R
    return matrix_pose(T)[3:]


def rotate_by(R, rotvec):
    return np.asarray(R, dtype=float) @ pose_matrix(np.r_[0., 0., 0., rotvec])[:3, :3]


def waypoint_plan(message):
    """Policy `waypoints` message (t_ns, poses = xyz + rot6d, gripper 0..1, per waypoint) -> (times int64,
    poses (N, 4, 4), gripper (N,), id)."""
    times = np.asarray(message["t_ns"], dtype=np.int64)
    poses, grip = np.asarray(message["poses"], dtype=float), np.asarray(message["gripper"], dtype=float)
    if (poses.ndim != 2 or poses.shape[1] != 9 or not len(times) == len(poses) == len(grip) > 0
            or not np.isfinite(poses).all() or not np.isfinite(grip).all() or (np.diff(times) <= 0).any()):
        raise ValueError("Invalid policy waypoints")
    T = np.tile(np.eye(4), (len(times), 1, 1))
    T[:, :3, :3], T[:, :3, 3] = rot6d_to_mat(poses[:, 3:]), poses[:, :3]
    return times, T, grip, message.get("id")


def waypoint_pose(times, poses, now_ns):
    """Pose at now_ns on timed waypoints (sorted times, (N, 4, 4) poses): linear in position, the shortest rotation
    in between; before the first / after the last waypoint: that waypoint (the target holds there)."""
    j = int(np.searchsorted(times, now_ns, side="right"))
    if j == 0 or j == len(times):
        return np.array(poses[0 if j == 0 else -1], dtype=float)
    T0, T1 = poses[j - 1], poses[j]
    u = (now_ns - times[j - 1]) / (times[j] - times[j - 1])
    out = np.eye(4)
    out[:3, 3] = (1 - u) * T0[:3, 3] + u * T1[:3, 3]
    out[:3, :3] = rotate_by(T0[:3, :3], u * relative_rotvec(T0[:3, :3], T1[:3, :3]))
    return out


def waypoint_value(times, values, now_ns):
    """The value of the last waypoint at or before now_ns (the first one before it): a step, not interpolated."""
    return float(values[max(int(np.searchsorted(times, now_ns, side="right")) - 1, 0)])


def approach(T, goal, max_m, max_rad):
    """T moved toward goal by at most max_m (translation) and max_rad (rotation, the shortest way): the speed limit
    on the policy target, so a jump (a new action chunk, or a slow reply) becomes a ramp."""
    T, goal = np.asarray(T, dtype=float), np.asarray(goal, dtype=float)
    out = T.copy()
    d = goal[:3, 3] - T[:3, 3]
    out[:3, 3] = T[:3, 3] + d * min(1., max_m / max(np.linalg.norm(d), 1e-12))
    rv = relative_rotvec(T[:3, :3], goal[:3, :3])
    out[:3, :3] = rotate_by(T[:3, :3], rv * min(1., max_rad / max(np.linalg.norm(rv), 1e-12)))
    return out


def send_json(node, name, obj):
    import pyarrow as pa
    node.send_output(name, pa.array([json.dumps(obj, allow_nan=False)]))


def read_json(event):
    return json.loads(event["value"][0].as_py())
