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
./teleop.sh labels.txt          # one label per line ('#' starts a comment): episode NNN gets line NNN
DRY_RUN=1 ./teleop.sh           # do not connect to the robot
OUT_DIR=/some/dir ./teleop.sh   # data location, default data/ in this directory
TASK=pick_cup ./teleop.sh       # skip the task question (continues pick_cup, or starts it if it is new)
```

Tasks: each task has its own folder `OUT_DIR/<task name>/`. At start-up, before the robot moves, teleop.sh lists the
existing tasks (most recently changed first, with their episode count and last episode) and asks:

```
Tasks in /data/tools/teleop/data:
  1  pick_cup  (12 episodes, last 012_episode)
  2  stack_blocks  (3 episodes, last 003_episode)
e = continue an existing task, n = new task:
```

- `e`, then the task's number key (1-9, no Enter; with 10 or more tasks type the number and Enter; with only one
  task it is continued directly).
- `n`, then type the name of the new task and Enter. Spaces and other characters become `_` (`pick cup` ->
  `pick_cup`); an existing name is refused. The new task starts at `001_episode`.

Episode numbers are per task and continue across runs: the terminal shows the task's last episode (with its length,
or INCOMPLETE if that run ended while recording) and the next one, e.g.
`Task pick_cup: last episode 012_episode (54 s, saved)` / `Next: 013_episode`. Pass the same label file again to
continue a plan.

The SpaceMouse (arm and gripper buttons) controls the robot only while an episode is being recorded: before Space,
while an episode is saved and the robot resets, and between episodes the robot does not follow it.

Keys: Space start / stop an episode; d then y delete the last episode of the task, also one of an earlier run (any
other key keeps it; the next episode takes its number and label again); h return home; Esc return home and quit;
Ctrl-C quit without homing; SpaceMouse left button opens / right button closes the gripper (press again to stop).
Start-up: preflight (robot, gripper, cameras) -> after 3 s the gripper opens and the robot returns home -> operate
once the status line shows ready. Every episode starts from this state: after each episode is saved the gripper
opens fully (an object still held is released where the arm is), then the robot returns home (moveJ), and Space
starts the next episode only once that is done (`HOME_AFTER_EPISODE` in the `recorder` node of `dataflow.yml`, '0'
turns it off). h and Esc return home without touching the gripper. The robot is locked while the cameras deliver
no frames.
Resolution and frame rate are set in the `recorder` node of `dataflow.yml`.

## Output

```
OUT_DIR/<task name>/
  <NNN>_<label>/
    episode.json               task, run, start / end time; per camera: frames, frames with pose, frame-number gaps, fps
    <role>.mp4                 RGB video (mp4v); frame i = row i of the CSV
    <role>_frames.csv          per frame: SDK frame number, timestamp, host time, frame_mono_ns, pose_valid,
                               tcp_x..tcp_qw, flange_x..flange_qw (base frame, metres, quaternion x y z w)
  _runs/<YYYYmmdd_HHMMSS>/     one teleop run (named by its start time); episode.json "run" points here
    session.json               camera serials + SDK RGB intrinsics, settings, the episodes this run recorded
    robot.jsonl                every ur5/state record of the run (125 Hz, unchanged)
OUT_DIR/_logs/                 teleop.sh log of each run
```

Pose per frame: frame time = SDK global_time (host system clock) mapped to CLOCK_MONOTONIC; the pose is
interpolated between the two surrounding robot records (position linear, rotation slerp). If a frame lies outside
the robot records or the two records are more than `MAX_POSE_GAP_MS` (50 ms) apart, `pose_valid=0` and the pose is
empty. `data/` and `out/` (dora node logs) are not committed.

## Tests (run in this directory, with the Python environment that has the requirements installed)

```bash
tests/smoke.sh                                          # SMOKE: the tool starts and shows its first prompt
python -m unittest tests.test_recorder tests.test_common tests.test_control \
    tests.test_keyboard tests.test_gripper tests.test_spacemouse tests.test_task   # UNIT
python -m unittest tests.test_integration               # INTEGRATION: full dataflow + teleop.sh task question
                                                        # (mock mouse / cameras, DRY_RUN, keys typed into a pty)
```

None of the tests connects to the robot or opens a real camera, and none needs the device settings in
`teleop.local.env`.
