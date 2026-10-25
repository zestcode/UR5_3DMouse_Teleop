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
STEPS="open pick place close" ./teleop.sh   # skip the steps question
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

Then the task's steps: the verbs of its atomic sub-skills in order, for AtomicVPP, which trains one expert per skill
(see "Atomic sub-skill segments"):

```
Steps of this task, verbs in order (e.g. open pick place close; Enter = no steps): open pick place close
  4 steps: open -> pick -> place -> close
```

A step is one word of letters (commas and `->` between them are fine too). A verb outside AtomicVPP's
`pick place turn open close` is accepted with a warning: add it to AtomicVPP's skill list before training. Enter
alone means no steps: Space starts and stops an episode, as before. The steps are asked once per task (also once
for a task from before steps existed) and kept in `task.json` as `"steps"`; `STEPS="..."` skips the question
(`STEPS=""`: no steps). With `TASK=...` and nothing known, the task has no steps.

Episode numbers are per task and continue across runs: the terminal shows the task's last episode (with its length,
or INCOMPLETE if that run ended while recording) and the next one, e.g.
`Task pick_cup: last episode 012_episode (54 s, saved)` / `Next: 013_episode`. Pass the same label file again to
continue a plan.

The SpaceMouse (arm and gripper buttons) controls the robot only while an episode is being recorded: before Space,
while an episode is saved and the robot resets, and between episodes the robot does not follow it.

Keys: Space start / stop an episode; Enter (a task with steps, while recording) end the current step, the last
Enter also stops the episode (the status line shows the current step, e.g. `step 2/4 pick`; Space before the last
step stops and saves the episode too, without segments: `2/4 steps: no segments` in the save message); d then y
delete the last episode of the task, also one of an earlier run (any
other key keeps it; the next episode takes its number and label again); h return home; Esc return home and quit;
Ctrl-C quit without homing; SpaceMouse left button opens / right button closes the gripper (press again to stop).
Start-up: preflight (robot, gripper, cameras) -> after 3 s the gripper opens and the robot returns home -> operate
once the status line shows ready. Every episode starts from this state: after each episode is saved the gripper
opens fully (an object still held is released where the arm is), then the robot returns home (moveJ), and Space
starts the next episode only once that is done (`HOME_AFTER_EPISODE` in the `recorder` node of `dataflow.yml`, '0'
turns it off). h and Esc return home without touching the gripper. The robot is locked while the cameras deliver
no frames.
A camera that delivers no frame for 1 s during an episode (`CAMERA_FAIL_S`) fails it: the episode is deleted (the
next one takes its number and label again), the gripper opens and the robot returns home as after a saved episode,
and the camera's stream is restarted (pipeline stop / start, then a hardware reset of the camera). Space waits until
every camera delivers frames again (`wrist camera delivers frames again`). A camera still without frames 30 s after
it stopped (`CAMERA_GIVE_UP_S`) ends the run once the robot is home; replug it and start teleop.sh again. Both are
in the `recorder` node of `dataflow.yml`. Any error of the recorder ends the whole run, so the terminal is never left
with a run whose robot no longer moves.
Resolution and frame rate are set in the `recorder` node of `dataflow.yml`.

## Output

```
OUT_DIR/<task name>/
  task.json                    {"instruction": "...", "steps": ["open", "pick", ...]}
  <NNN>_<label>/
    episode.json               task, run, instruction, start / end time; per camera: frames, frames with pose,
                               frame-number gaps, frames retimed (see below), fps; steps and
                               step_end_host_mono_ns (the time of each Enter, CLOCK_MONOTONIC as frame_mono_ns)
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
  videos/{train,val}/<id>/0.mp4       fixed camera (VPP rgb_static), cropped (VPP_CROP), 256 x 256, 10 fps, H.264
  videos/{train,val}/<id>/1.mp4       wrist camera (VPP rgb_gripper)
  latent_videos/{train,val}/<id>/<k>.pt   SVD VAE latents: written by vpp_latents.py (below)
  segments.json                       atomic sub-skill segments of the episodes with steps (AtomicVPP, below)
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
  pose with servoL as the ur5 node does, and prepare the camera images with `vpp.square(image, 256, crop)`, with
  `crop` from the annotation's `crop` for that camera (none: the whole frame), so they get the same crop and resize.
- Crop: `VPP_CROP` (`role:x0,y0,x1,y1 ...`, i.e. `frame[y0:y1, x0:x1]` in pixels of the 640 x 480 frame, x1 and
  y1 exclusive) cuts a camera's frames before the pad and resize; default `fixed:150,100,500,450`, a 350 x 350 box
  around the table and the arm. A square box is not distorted by the square resize. To choose one: open
  `crop_region.ipynb`, set its video and box, and check the box on frames across the episode.
- Settings: `VPP_*` in the `recorder` node of `dataflow.yml` (`VPP_CAMERAS: ''` turns the export off). Deleting an
  episode (d, y) deletes its export too. Episodes that ended by an error are not exported.
- Re-export (other settings, a corrected instruction, or episodes recorded before the export existed):
  `python vpp.py OUT_DIR/<task> [episode names] [--instruction "..."] [--crop "fixed:150,100,500,450"]
  [--val-every 10] [--out DIR]` (`--crop ''`: whole frames). A re-export also deletes the episode's latents.

### Atomic sub-skill segments

AtomicVPP's MoE policies route each training window to the expert of its skill, from a segments file in the shape
of its `libero_cosmos_segments.json`. The export writes one for the episodes of a task with steps,
`<task>/_vpp/segments.json`, keyed by episode id:

```
{"1": {"all_frames": 31, "segments": [{"start_frame": 0, "end_frame": 9, "primary_action_verb": "open"},
                                      {"start_frame": 10, "end_frame": 18, "primary_action_verb": "pick"}, ...]}}
```

- Frames are the 10 Hz steps of the export (`all_frames` = the annotation's `video_length`). A step at time t
  belongs to the step whose Enter is the first at or after t; the steps after the last Enter belong to the last
  step. So the segments are contiguous and cover every frame, as AtomicVPP's loader requires. A step shorter than
  0.1 s has no segment.
- The annotation of the episode has the same list as `segments`.
- Only episodes whose every step was ended with Enter get segments (one stopped early with Space does not).
  Deleting an episode (d, y) or re-exporting it updates the file; an episode re-exported with `python vpp.py` gets
  its segments from its `episode.json`.
- AtomicVPP's dataset code and skill list are for LIBERO: training on the UR5 needs a UR5 copy of
  `policy_models/datasets/libero_dataset_moe.py` that reads this file, and the skill list
  (`policy_models/utils/libero_skill_mapping.py`) with the verbs used here.

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

## Running the VPP policy on the robot

The trained policy runs in a server process (`step3_deploy_real_ur5.py` of `AtomicVPP_LIBERO`, conda env `vpp`),
by default on this machine's GPU (a minimal copy in `../vpp`, see its README), or on pegasus. The dataflow records
the cameras and the robot in real time, sends one observation per policy step and executes the action chunks it
gets back:

```
dataflow (deploy.yml)                                           policy server (step3_deploy_real_ur5.py)
  cameras -> policy.py: latest frame of each camera,
             vpp.square (fixed view cropped FIRST, then           VAE + SVD features + action head
             resized to 256), BGR -> RGB, state at frame time  -> 10 targets (xyz, rot6d, gripper), 0.1 s apart
  ur5.py (CONTROL=policy) <- timed waypoints <- policy.py  <--  (absolute: chunk + state at the frame time)
```

```bash
./deploy.sh                     # asks for the task (its instruction conditions the policy), then as teleop.sh
DRY_RUN=1 ./deploy.sh           # pipeline check: real cameras and server, no robot (the state is simulated)
```

- `deploy.sh` checks the server and that its training data had the cameras, crop, size and fps of `deploy.yml`
  (`python policy.py --check`), and starts it if none answers (`deploy_ur5_server.sh start`; it keeps running for
  later runs). Where, `VPP_HOST`:
  - `local` (default): this machine's GPU, from `../vpp` with the conda env `vpp`. Measured on the RTX 4070 Ti:
    6.5 GB of its 12 GB, 1.4 GB of RAM once loaded (about 7 GB more while loading), ~30 s to load. Stop it to
    free them: `bash ../vpp/deploy_ur5_server.sh stop`.
  - an ssh host, e.g. `pegasus` (key login): `deploy.sh` opens an SSH tunnel to it and starts the server in
    `/mnt/raptor/kim/AtomicVPP_LIBERO` on its least used GPU (~7.5 GB, ~1-5 min to load;
    `ssh pegasus bash /mnt/raptor/kim/AtomicVPP_LIBERO/deploy_ur5_server.sh stop`).
  - `''`: start nothing, the server already answers at `POLICY_ADDR`.

  Server settings: `VPP_*` in `teleop.local.env` (see the example); default checkpoint
  `ur5_dp/checkpoints/last.pt` with its EMA weights (lower val error than the raw weights in that run).
- Keys: Space starts a rollout (the policy moves the robot) and stops it (the arm holds where it is, then the
  gripper opens and the robot returns home); h stops it and returns home; Esc returns home and quits. A rollout
  also stops after `ROLLOUT_MAX_S` (60 s), and when the server fails or answers later than `REQUEST_TIMEOUT_S`.
  Keep the emergency stop at hand.
- Motion: the ur5 node follows the chunk's targets at their times (interpolated at 125 Hz), never faster than
  `MAX_LINEAR_M_S` / `MAX_ANGULAR_RAD_S`, inside the same workspace box and leash as teleop (`deploy.yml` has the
  ur5 settings of `dataflow.yml`; a unit test keeps them equal); the gripper closes / opens when the chunk's
  command crosses 0.5. Replanning (`REPLAN_STEPS: '0'`): the next observation is taken so its chunk arrives just
  before the current one ends (frame to chunk ~225 ms with the server on this machine's RTX 4070 Ti, inference
  ~180 ms while the cameras record; round trip ~265 ms on a pegasus V100 through the tunnel), i.e. about every
  0.6 s.
- Recorded, per run, in `OUT_DIR/<task>/_policy/<run>/`: `session.json` (the served checkpoint, its val metrics,
  the settings), `robot.jsonl`, and each rollout `<NNN>_rollout/` as a teleop episode (`<role>.mp4`,
  `<role>_frames.csv`, `episode.json`) plus `policy.jsonl` (per step: frame times, state, chunk, latency) and
  `policy_<role>.mp4` (the 256 x 256 images the policy was given).
- Before the first run on the robot, check the whole path except the robot on a recorded val episode:
  `python policy_check.py OUT_DIR/<task>/010_episode` (with the server running: `bash ../vpp/deploy_ur5_server.sh
  start`, or for pegasus with the tunnel open, e.g. while `deploy.sh` runs, or
  `ssh -N -L 47110:127.0.0.1:47110 pegasus &`). It prepares the raw frames as the live node does and compares the
  images with the export's frames (a mean difference of a few levels: H.264; ~20 or more: another crop or colour
  order), the state with its states, and the chunks with the operator's actions (about the training val error).

## Tests (run in this directory, with the Python environment that has the requirements installed)

```bash
tests/smoke.sh                                          # SMOKE: the tool starts and shows its first prompt
python -m unittest tests.test_recorder tests.test_common tests.test_control \
    tests.test_keyboard tests.test_gripper tests.test_spacemouse tests.test_task tests.test_vpp \
    tests.test_policy                                   # UNIT
                                                        # (test_vpp: also vpp_preview.py; expects ffmpeg with libx264)
python -m unittest tests.test_vpp_latents               # UNIT, needs torch: skipped without it (stub VAE)
python -m unittest tests.test_integration               # INTEGRATION: full dataflow + teleop.sh task question
                                                        # (mock mouse / cameras, DRY_RUN, keys typed into a pty)
python -m unittest tests.test_deploy                    # INTEGRATION: policy dataflow + deploy.sh against a fake
                                                        # policy server (tests/fake_policy_server.py)
```

None of the tests connects to the robot, opens a real camera or needs the policy server, and none needs the
device settings in `teleop.local.env`. `policy_check.py` (above) is the check against the real server.
