# teleop - UR5 SpaceMouse teleoperation + RGB video + per-frame pose

Self-contained tool: teleoperate a UR5 (Robotiq gripper) with a SpaceMouse, record the RGB video of each RealSense
D455, and attach the robot pose to every video frame. RGB (color stream) only: no IR, no depth, no camera warm-up.
All paths are relative to this directory, so the whole directory can be copied to another machine.

## Setup on a new machine

```bash
git clone <repo> teleop && cd teleop
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # tested with Python 3.12
cp teleop.local.env.example teleop.local.env                          # fill in ROBOT_IP, CAMERAS, TELEOP_VENV
```

- `teleop.local.env` (not committed): robot IP, cameras as `role:serial` (list serials with `rs-enumerate-devices -s`),
  and the Python environment path. If a value is missing or still a placeholder, the preflight stops with
  "not configured" and connects to nothing.
- System dependencies: `spacenavd` (SpaceMouse daemon, must be running), the librealsense udev rules, and the
  Robotiq URCap installed on the UR5 and activated on the teach pendant.
- Arm-specific parameters (home joint angles, workspace z floor, speed limits, gripper speed / force) are in the
  `ur5` node of `dataflow.yml` and were set for the current robot; change them first for another arm or mounting.

## Usage

```bash
./teleop.sh                     # episodes are named 001_episode, 002_episode, ...
./teleop.sh labels.txt          # one label per line ('#' starts a comment) names the episodes
DRY_RUN=1 ./teleop.sh           # do not connect to the robot
OUT_DIR=/some/dir ./teleop.sh   # data location, default data/ in this directory
```

Keys: Space start / stop an episode; h return home; Esc return home and quit; Ctrl-C quit without homing;
SpaceMouse left button opens / right button closes the gripper (press again to stop).
Start-up: preflight (robot, gripper, cameras) -> the robot returns home after 3 s -> operate once the status line
shows ready. The robot is locked while the cameras deliver no frames.
Resolution and frame rate are set in the `recorder` node of `dataflow.yml`.

## Output

```
OUT_DIR/<date_time_id>/
  session.json                 camera serials + SDK RGB intrinsics, settings, list of episodes
  robot.jsonl                  every ur5/state record of the run (125 Hz, unchanged)
  <NNN>_<label>/
    episode.json               start / end time; per camera: frames, frames with pose, frame-number gaps, fps
    <role>.mp4                 RGB video (mp4v); frame i = row i of the CSV
    <role>_frames.csv          per frame: SDK frame number, timestamp, host time, frame_mono_ns, pose_valid,
                               tcp_x..tcp_qw, flange_x..flange_qw (base frame, metres, quaternion x y z w)
```

Pose per frame: frame time = SDK global_time (host system clock) mapped to CLOCK_MONOTONIC; the pose is
interpolated between the two surrounding robot records (position linear, rotation slerp). If a frame lies outside
the robot records or the two records are more than `MAX_POSE_GAP_MS` (50 ms) apart, `pose_valid=0` and the pose is
empty. `data/` and `out/` (dora node logs) are not committed.

## Tests (run in this directory, with the Python environment that has the requirements installed)

```bash
tests/smoke.sh                                          # SMOKE: the tool starts and shows its first prompt
python -m unittest tests.test_recorder tests.test_common tests.test_control \
    tests.test_keyboard tests.test_gripper tests.test_spacemouse   # UNIT
python -m unittest tests.test_integration               # INTEGRATION: full dataflow (mock mouse / cameras, DRY_RUN)
```

None of the tests connects to the robot or opens a real camera, and none needs the device settings in
`teleop.local.env`.
