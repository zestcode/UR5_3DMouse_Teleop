"""UNIT tests of task.py (the start-up task question) on its own: name cleaning, the task list, and the questions
(task, then its instruction), answered through a pseudo-terminal (no dora, no robot, no cameras)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import task as t
from tests.ptyutil import PtyProcess

HERE = Path(__file__).resolve().parents[1]


def make_tasks(root, spec, files=()):
    """spec: {task: [episode dir, ...]}, oldest task first (mtimes set in that order); files: extra plain files."""
    for i, (name, episodes) in enumerate(spec.items()):
        for e in episodes:
            (root / name / e).mkdir(parents=True)
        (root / name).mkdir(parents=True, exist_ok=True)
        for f in files:
            if f.startswith(name + "/"):
                (root / f).write_text("")
        os.utime(root / name, (1_000_000 + i, 1_000_000 + i))  # after the contents: they change the mtime


class TestNames(unittest.TestCase):
    def test_clean_name(self):
        self.assertEqual(t.clean_name("  pick cup "), "pick_cup")
        self.assertEqual(t.clean_name("a/b?c"), "a_b_c")
        self.assertEqual(t.clean_name("_logs"), "logs")  # never a hidden / reserved folder
        self.assertEqual(t.clean_name("拿 杯子"), "拿_杯子")
        self.assertEqual(t.clean_name(" ./ "), "")

    def test_task_list_most_recent_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(t.tasks(root / "missing"), [])
            make_tasks(root, {"old": ["001_episode", "002_episode", "010_episode", "_runs", "notes"],
                              "_logs": [], ".hidden": [], "new": []},
                       files=["old/011_file"])  # a file, not an episode
            self.assertEqual(t.tasks(root), [("new", 0, None), ("old", 3, "010_episode")])


class TestQuestions(unittest.TestCase):
    def ask(self, root, env_extra=None):
        env = {k: v for k, v in os.environ.items() if k != "TASK"}
        return PtyProcess([sys.executable, str(HERE / "task.py")], {**env, "OUT_DIR": str(root), **(env_extra or {})},
                          stdout=subprocess.PIPE)

    def result(self, p, instruction=None):
        """-> (exit status, stdout: task name and instruction); instruction: typed at its question first."""
        if instruction is not None:
            p.wait_for("Instruction for this task, one English sentence")
            p.key(instruction + b"\n")
        out = p.proc.communicate(timeout=10)[0].decode().strip()
        return p.proc.returncode, out

    def test_no_tasks_asks_for_a_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.ask(Path(tmp))
            at = p.wait_for("No tasks in")
            p.wait_for("Name of the new task: ", at)
            p.key(b"pick cup\n")
            p.wait_for('(Enter = "pick cup"): ')
            self.assertEqual(self.result(p, b""), (0, "pick_cup\npick cup"))  # Enter: the name
            p.wait_for("folder name: pick_cup")
            self.assertEqual(list(Path(tmp).iterdir()), [])  # the recorder creates the folder, not task.py

    def test_e_with_one_task_continues_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_tasks(Path(tmp), {"pick": ["001_episode"]})
            p = self.ask(Path(tmp))
            p.wait_for("e = continue an existing task, n = new task: ")
            p.key(b"x")  # other keys are ignored
            p.key(b"e")
            self.assertEqual(self.result(p, b"pick up the red block"), (0, "pick\npick up the red block"))

    def test_e_with_several_tasks_one_number_key_picks(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_tasks(Path(tmp), {"older": [], "newer": ["001_episode"]})
            p = self.ask(Path(tmp))
            at = p.wait_for("1  newer  (1 episode, last 001_episode)")
            at = p.wait_for("2  older  (0 episodes)", at)
            p.key(b"e")
            p.wait_for("Press the task number (1-2): ", at)
            p.key(b"9")  # not a listed number: ignored
            p.key(b"2")  # no Enter
            self.assertEqual(self.result(p, b""), (0, "older\nolder"))

    def test_e_with_ten_or_more_tasks_takes_number_and_enter(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_tasks(Path(tmp), {f"task_{i}": [] for i in range(1, 12)})  # task_11 newest = 1, task_1 = 11
            p = self.ask(Path(tmp))
            at = p.wait_for("11  task_1  (0 episodes)")
            p.key(b"e")
            at = p.wait_for("Task number (1-11), then Enter: ", at)
            p.key(b"12\n")
            p.wait_for("not one of the task numbers", at)
            p.key(b"11\n")
            self.assertEqual(self.result(p, b""), (0, "task_1\ntask 1"))

    def test_n_refuses_an_existing_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_tasks(Path(tmp), {"pick": []})
            p = self.ask(Path(tmp))
            p.wait_for("e = continue")
            p.key(b"n")
            at = p.wait_for("Name of the new task: ")
            p.key(b"pick\n")
            at = p.wait_for("task pick already exists", at)
            p.key(b"???\n")
            at = p.wait_for("at least one letter or digit", at)
            p.key(b"place\n")
            self.assertEqual(self.result(p, b"  place it  "), (0, "place\nplace it"))

    def test_instruction_of_task_json_is_not_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_tasks(Path(tmp), {"pick": ["001_episode"]})
            (Path(tmp) / "pick" / "task.json").write_text(json.dumps({"instruction": "pick up the cup"}))
            p = self.ask(Path(tmp))
            p.wait_for("e = continue")
            p.key(b"e")
            p.wait_for("Instruction: pick up the cup")
            self.assertEqual(self.result(p), (0, "pick\npick up the cup"))
            p = self.ask(Path(tmp), {"INSTRUCTION": "lift the cup"})  # INSTRUCTION: no question either
            p.wait_for("e = continue")
            p.key(b"e")
            self.assertEqual(self.result(p), (0, "pick\nlift the cup"))

    def test_task_env_skips_the_questions_and_no_terminal_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items() if k != "INSTRUCTION"}
            env = {**env, "OUT_DIR": tmp, "TASK": "my task"}
            run = lambda env: subprocess.run([sys.executable, str(HERE / "task.py")], env=env, stdin=subprocess.DEVNULL,
                                             capture_output=True, text=True, timeout=10)
            out = run(env)
            self.assertEqual((out.returncode, out.stdout.strip()), (0, "my_task\nmy task"))  # instruction: the name
            (Path(tmp) / "my_task").mkdir()
            (Path(tmp) / "my_task" / "task.json").write_text(json.dumps({"instruction": "do it"}))
            self.assertEqual(run(env).stdout.strip(), "my_task\ndo it")
            self.assertEqual(run({**env, "INSTRUCTION": "do that"}).stdout.strip(), "my_task\ndo that")
            env.pop("TASK")
            out = subprocess.run([sys.executable, str(HERE / "task.py")], env=env, stdin=subprocess.DEVNULL,
                                 capture_output=True, text=True, timeout=10)
            self.assertNotEqual(out.returncode, 0)
            self.assertIn("set TASK=", out.stderr)


if __name__ == "__main__":
    unittest.main()
