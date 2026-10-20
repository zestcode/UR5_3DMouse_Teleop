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
- SpaceMouse feel is in the `spacemouse` node of `dataflow.yml`: `MOUSE_MAX` (full-scale deflection), `DEADZONE`,
  and for rotation alone `ROTATION_DEADZONE`, `ROTATION_CURVE` (exponent; 2 = finer near the centre, same top
  speed) and `ROTATION_COUPLING` (pushing the cap also tilts it, so while translating a twist must be firmer to
  rotate; 0 turns this off). Top speeds are `MAX_LINEAR_M_S` / `MAX_ANGULAR_RAD_S` in the `ur5` node.

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

Then the task's instruction: one English sentence saying what the robot does (`put both blocks into the bowl`),
used as the language condition by VPP and other language-conditioned policies. Enter takes the task name with
spaces. It is asked once per task and kept in `<task>/task.json` (edit that file to change it for later episodes;
`python vpp.py <task dir> --instruction "..."` rewrites it in the VPP export of earlier ones). `INSTRUCTION="..."`
skips the question.

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
  task.json                    {"instruction": "..."}
  <NNN>_<label>/
    episode.json               task, run, instruction, start / end time; per camera: frames, frames with pose,
                               frame-number gaps, frames retimed (see below), fps
    <role>.mp4                 RGB video (mp4v, MPEG-4 Part 2: plays in VLC / ffplay, not in browsers or VS Code;
                               the VPP export and its preview are H.264); frame i = row i of the CSV
    <role>_frames.csv          per frame: SDK frame number, timestamp, host time, frame_mono_ns, pose_valid,
                               tcp_x..tcp_qw, flange_x..flange_qw (base frame, metres, quaternion x y z w),
                               frame_time_source (sdk / receive)
  _runs/<YYYYmmdd_HHMMSS>/     one teleop run (named by its start time); episode.json "run" points here
    session.json               camera serials + SDK RGB intrinsics, settings, the episodes this run recorded
    robot.jsonl                every ur5/state record of the run (125 Hz, unchanged)
  _vpp/                        the same episodes in the dataset layout of VPP (see "Training VPP")
OUT_DIR/_logs/                 teleop.sh log of each run
```

Pose per frame: frame time = SDK global_time (host system clock) mapped to CLOCK_MONOTONIC; the pose is
interpolated between the two surrounding robot records (position linear, rotation slerp). The SDK time is not
always right: it has been seen up to 3.9 s ahead of the moment the frame reached the host, for a minute. A frame
whose SDK time is after its receive time, or more than 100 ms before it (normal: 10-18 ms), gets receive time
minus the camera's median latency in that episode instead (`frame_time_source = receive`; `timestamp_ms` keeps
the SDK value; the save message counts these frames). The quaternion has w >= 0, so it can jump between q and -q
when w is near 0 (the gripper pointing down); use the rot6d of robot.jsonl or the VPP export for learning. If a frame lies outside
the robot records or the two records are more than `MAX_POSE_GAP_MS` (50 ms) apart, `pose_valid=0` and the pose is
empty. `data/` and `out/` (dora node logs) are not committed.

## Training VPP

[Video Prediction Policy](https://github.com/roboterax/video-prediction-policy) does not read the raw episodes:
it needs, per episode, 10 Hz frames of every camera at the same steps, 256 x 256 videos, the state and action
of each step, and the instruction. The recorder writes this for every saved episode (in the background, a few
seconds; `VPP: <episode> exported` on the terminal) to `<task>/_vpp/`, in the layout VPP's
`step1_prepare_latent.py` produces for its real robots:

```
<task>/_vpp/
  annotation/{train,val}/<id>.json    texts, videos, latent_videos, episode_id, video_length, states, actions
  videos/{train,val}/<id>/0.mp4       fixed camera (VPP rgb_static), 256 x 256, 10 fps, H.264
  videos/{train,val}/<id>/1.mp4       wrist camera (VPP rgb_gripper)
  latent_videos/{train,val}/<id>/<k>.pt   SVD VAE latents: written by vpp_latents.py (below)
  preview/<id>.mp4                    optional, vpp_preview.py: both cameras side by side (not read by VPP)
```

The videos are H.264 when the `ffmpeg` command has libx264 (`ffmpeg -encoders | grep libx264`; Ubuntu's ffmpeg
has it), otherwise OpenCV's MPEG-4 Part 2, which VPP reads too but browsers and VS Code do not play
(`video_codec` in the annotation says which).

To check the synchronization by eye: `python vpp_preview.py OUT_DIR/<task>/_vpp [episode ids]` writes
`_vpp/preview/<id>.mp4`: per step both cameras side by side, with the step, its time, each camera's raw frame
number (its CSV row) and the gripper position / target, in real time (10 fps).

- `<id>` is the episode number (`007_episode` -> 7); episode numbers divisible by 10 go to `val`.
- Steps are 0.1 s apart, over the time all cameras recorded. Each camera contributes its frame nearest to the
  step (at most half a frame period away; `source.max_offset_ms` in the annotation), so the cameras line up even
  when one of them runs slower.
- `states` (10 per step): TCP xyz (m, base frame), TCP rot6d (first two rows of R), gripper position (0 open .. 1
  closed). `actions` (10): the servo target pose the SpaceMouse commanded (xyz + rot6d) and the gripper target
  (1 closing or holding an object, 0 opening, its position when stopped with the button). VPP trains on
  `actions[t..t+n] - states[t]`, so both are in the same space. When the policy runs, send the predicted target
  pose with servoL as the ur5 node does, and prepare the camera images with `vpp.square()` (the same resize).
- Settings: `VPP_*` in the `recorder` node of `dataflow.yml` (`VPP_CAMERAS: ''` turns the export off). Deleting an
  episode (d, y) deletes its export too. Episodes that ended by an error are not exported.
- Re-export (other settings, a corrected instruction, or episodes recorded before the export existed):
  `python vpp.py OUT_DIR/<task> [episode names] [--instruction "..."] [--val-every 10] [--out DIR]`.

Then, on the GPU machine, in VPP's own Python environment:

1. Latents: `python vpp_latents.py <svd-robot folder> OUT_DIR/<task>/_vpp` (skips videos that already have one).
2. Video model: in `video_conf/train_svd.yaml` set `dataset_dir: OUT_DIR` and `dataset: '<task>/_vpp'`
   (several tasks: joined with `+`, with `prob`). VPP chooses some behaviour from words in the path: with
   `xhand` or `calvin` in it, it also trains on the second camera; `cloth`, `bridge`, `rt1`, `sthv2` change the
   frame sampling.
3. Action model: `step2_prepare_json.py` (dataset names, data root) writes the sample list and the state / action
   percentiles for `policy_conf/VPP_xbot_train.yaml`. The x-bot code is written for VPP's two-arm robot with
   hands (38 values, 3 cameras), so adapt it for 10 values and 2 cameras: `action_dim` / `proprio_dim: 10`, the
   loss slices in `policy_models/VPP_policy_xbot.py` (xyz `0:3`, rotation `3:9`, gripper `9:10`), and drop the
   third camera (`rgb_gripper2`) in `policy_models/datasets/xbot_dataset.py` (the policy works without it).

## Tests (run in this directory, with the Python environment that has the requirements installed)

```bash
tests/smoke.sh                                          # SMOKE: the tool starts and shows its first prompt
python -m unittest tests.test_recorder tests.test_common tests.test_control \
    tests.test_keyboard tests.test_gripper tests.test_spacemouse tests.test_task tests.test_vpp   # UNIT
                                                        # (test_vpp: also vpp_preview.py; expects ffmpeg with libx264)
python -m unittest tests.test_vpp_latents               # UNIT, needs torch: skipped without it (stub VAE)
python -m unittest tests.test_integration               # INTEGRATION: full dataflow + teleop.sh task question
                                                        # (mock mouse / cameras, DRY_RUN, keys typed into a pty)
```

None of the tests connects to the robot or opens a real camera, and none needs the device settings in
`teleop.local.env`.
