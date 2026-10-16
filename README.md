# teleop — UR5 SpaceMouse 遥操 + RGB 视频 + 每帧位姿

独立工具包：SpaceMouse 遥操 UR5（Robotiq 夹爪），录每台 RealSense D455 的 RGB 视频，并给每一帧配上机器人位姿。
只录 RGB（color 流），不开 IR / depth，没有相机预热。所有路径都相对本目录，可整个目录拷到其他机器使用。

## 新机器部署

```bash
git clone <repo> teleop && cd teleop
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.12 测试过
cp teleop.local.env.example teleop.local.env                          # 填 ROBOT_IP、CAMERAS、TELEOP_VENV
```

- `teleop.local.env`（不提交）：机器人 IP、相机 `角色:序列号`（`rs-enumerate-devices -s` 可查）、Python 环境路径。
  没填或还是占位值时，预检直接报 “not configured”，不会连接任何设备。
- 系统依赖：`spacenavd`（SpaceMouse 守护进程，需在运行）、librealsense 的 udev 规则、UR5 上装好 Robotiq URCap 并在示教器上 Activate。
- 机械臂相关参数（home 关节角、工作空间 z 下限、速度上限、夹爪速度/力）在 `dataflow.yml` 的 `ur5` 节点里，
  是按当前这台机器标定的；换一台机械臂或改了安装，先改这里。

## 用法

```bash
./teleop.sh                     # episode 命名 001_episode, 002_episode …
./teleop.sh labels.txt          # 每行一个标签（# 起注释）给 episode 命名
DRY_RUN=1 ./teleop.sh           # 不连机器人
OUT_DIR=/某目录 ./teleop.sh     # 数据位置，默认本目录下 data/
```

按键：Space 开始/停止 episode · h 回 home · Esc 回 home 并退出 · Ctrl-C 直接退出 · 鼠标左键夹爪开 / 右键夹爪合（再按一次停）。
启动顺序：预检（机器人、夹爪、相机）→ 3 s 后自动回 home → 状态行显示就绪后可操作。相机不出帧时机器人锁定。
分辨率、帧率等在 `dataflow.yml` 的 `recorder` 节点里改。

## 输出

```
OUT_DIR/<日期_时间_id>/
  session.json                 相机序列号 + SDK RGB 内参、设置、episode 列表
  robot.jsonl                  整个运行期间每条 ur5/state（125 Hz，原样）
  <NNN>_<label>/
    episode.json               起止时间、每台相机帧数 / 有位姿帧数 / 丢帧数 / fps
    <role>.mp4                 RGB 视频（mp4v），第 i 帧 = CSV 第 i 行
    <role>_frames.csv          每帧：SDK 帧号、时间戳、主机时间、frame_mono_ns、pose_valid、
                               tcp_x..tcp_qw、flange_x..flange_qw（基座系，米，四元数 x y z w）
```

位姿对应：帧时间 = SDK global_time（主机系统时钟）换到 CLOCK_MONOTONIC；在前后两条机器人记录间插值（位置线性、
旋转 slerp）；帧不在机器人记录范围内或前后间隔 > `MAX_POSE_GAP_MS`（50 ms）时 `pose_valid=0`、位姿为空。
`data/`、`out/`（dora 节点日志）不提交。

## 测试（在本目录下，用装好依赖的 Python 环境）

```bash
tests/smoke.sh                                          # SMOKE：工具能否启动并出现第一条提示
python -m unittest tests.test_recorder tests.test_common tests.test_control \
    tests.test_keyboard tests.test_gripper tests.test_spacemouse   # UNIT
python -m unittest tests.test_integration               # INTEGRATION：整条 dataflow（模拟鼠标/相机，DRY_RUN）
```

三类测试都不连机器人、不开真相机，不需要 `teleop.local.env` 里的设备信息。
