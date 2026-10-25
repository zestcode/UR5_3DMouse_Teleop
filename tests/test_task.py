"""UNIT tests of task.py (the start-up task question) on its own: name cleaning, the task list, step verbs, and the
questions (task, then its instruction, then with --steps its steps), answered through a pseudo-terminal (no dora, no
robot, no cameras)."""
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


class TestSteps(unittest.TestCase):
    def test_parse_steps(self):
        self.assertEqual(t.parse_steps("open pick place close"), ["open", "pick", "place", "close"])
        self.assertEqual(t.parse_steps(" Open, PICK -> place "), ["open", "pick", "place"])  # as the summary shows them
        self.assertEqual(t.parse_steps("  "), [])

    def test_a_step_is_one_word_of_letters(self):
        for text in ("pick2", "pick_up", "pick-up", "放"):
            with self.assertRaises(ValueError, msg=text):
                t.parse_steps(text)

    def test_read_steps_tells_never_asked_from_no_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp)
            self.assertIsNone(t.read_steps(task / "missing"))
            (task / "task.json").write_text(json.dumps({"instruction": "x"}))
            self.assertIsNone(t.read_steps(task))  # a task from before steps existed: ask
            (task / "task.json").write_text(json.dumps({"instruction": "x", "steps": []}))
            self.assertEqual(t.read_steps(task), [])  # answered: no steps
            (task / "task.json").write_text(json.dumps({"instruction": "x", "steps": ["open", "pick"]}))
            self.assertEqual(t.read_steps(task), ["open", "pick"])


class TestQuestions(unittest.TestCase):
    def ask(self, root, env_extra=None, args=()):
        env = {k: v for k, v in os.environ.items() if k not in ("TASK", "STEPS")}
        return PtyProcess([sys.executable, str(HERE / "task.py"), *args],
                          {**env, "OUT_DIR": str(root), **(env_extra or {})}, stdout=subprocess.PIPE)

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

    def test_new_task_asks_its_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.ask(Path(tmp), args=["--steps"])
            p.wait_for("Name of the new task: ")
            p.key(b"drawer\n")
            p.wait_for("Instruction for this task")
            p.key(b"put the cup in the drawer\n")
            at = p.wait_for("Steps of this task, verbs in order (e.g. open pick place close; Enter = no steps): ")
            p.key(b"open pick2\n")
            at = p.wait_for("pick2", at)  # refused: asked again
            at = p.wait_for("Steps of this task", at)
            p.key(b"Open pick push close\n")
            at = p.wait_for("push", at)  # a verb AtomicVPP does not know: a warning, still taken
            p.wait_for("4 steps: open -> pick -> push -> close", at)
            self.assertEqual(self.result(p), (0, "drawer\nput the cup in the drawer\nopen pick push close"))

    def test_enter_at_the_steps_question_means_no_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.ask(Path(tmp), args=["--steps"])
            p.wait_for("Name of the new task: ")
            p.key(b"pick\n")
            p.wait_for("Instruction for this task")
            p.key(b"\n")
            p.wait_for("Steps of this task")
            p.key(b"\n")
            p.wait_for("no steps")
            self.assertEqual(self.result(p), (0, "pick\npick"))  # the empty third line

    def test_steps_of_task_json_are_not_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_tasks(Path(tmp), {"pick": ["001_episode"]})
            for steps, shown, out in ((["open", "pick"], "Steps: open -> pick", "open pick"), ([], "Steps: none", "")):
                (Path(tmp) / "pick" / "task.json").write_text(json.dumps({"instruction": "x", "steps": steps}))
                p = self.ask(Path(tmp), args=["--steps"])
                p.wait_for("e = continue")
                p.key(b"e")
                p.wait_for(shown)
                self.assertEqual(self.result(p), (0, f"pick\nx\n{out}".strip()))

    def test_existing_task_without_steps_is_asked_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_tasks(Path(tmp), {"pick": ["001_episode"]})
            (Path(tmp) / "pick" / "task.json").write_text(json.dumps({"instruction": "x"}))  # before steps existed
            p = self.ask(Path(tmp), args=["--steps"])
            p.wait_for("e = continue")
            p.key(b"e")
            p.wait_for("Steps of this task")
            p.key(b"pick place\n")
            self.assertEqual(self.result(p), (0, "pick\nx\npick place"))
            p = self.ask(Path(tmp), {"STEPS": "turn"}, args=["--steps"])  # STEPS: no question either
            p.wait_for("e = continue")
            p.key(b"e")
            self.assertEqual(self.result(p), (0, "pick\nx\nturn"))

    def test_without_steps_option_no_steps_question(self):
        with tempfile.TemporaryDirectory() as tmp:  # deploy.sh: two lines, as before steps existed
            make_tasks(Path(tmp), {"pick": ["001_episode"]})
            (Path(tmp) / "pick" / "task.json").write_text(json.dumps({"instruction": "x", "steps": ["pick"]}))
            p = self.ask(Path(tmp))
            p.wait_for("e = continue")
            p.key(b"e")
            self.assertEqual(self.result(p), (0, "pick\nx"))
            self.assertNotIn(b"Steps", p.out)

    def test_task_env_skips_the_questions_and_no_terminal_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items() if k not in ("INSTRUCTION", "STEPS")}
            env = {**env, "OUT_DIR": tmp, "TASK": "my task"}
            run = lambda env: subprocess.run([sys.executable, str(HERE / "task.py")], env=env, stdin=subprocess.DEVNULL,
                                             capture_output=True, text=True, timeout=10)
            out = run(env)
            self.assertEqual((out.returncode, out.stdout.strip()), (0, "my_task\nmy task"))  # instruction: the name
            (Path(tmp) / "my_task").mkdir()
            (Path(tmp) / "my_task" / "task.json").write_text(json.dumps({"instruction": "do it"}))
            self.assertEqual(run(env).stdout.strip(), "my_task\ndo it")
            self.assertEqual(run({**env, "INSTRUCTION": "do that"}).stdout.strip(), "my_task\ndo that")
            steps = lambda env: subprocess.run([sys.executable, str(HERE / "task.py"), "--steps"], env=env,
                                               stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10)
            self.assertEqual(steps(env).stdout, "my_task\ndo it\n\n")  # no steps known: none, not asked
            self.assertEqual(steps({**env, "STEPS": "Pick, place"}).stdout, "my_task\ndo it\npick place\n")
            (Path(tmp) / "my_task" / "task.json").write_text(json.dumps({"instruction": "do it", "steps": ["turn"]}))
            self.assertEqual(steps(env).stdout, "my_task\ndo it\nturn\n")
            out = steps({**env, "STEPS": "pick2"})
            self.assertNotEqual(out.returncode, 0)
            self.assertIn("STEPS", out.stderr)
            env.pop("TASK")
            out = subprocess.run([sys.executable, str(HERE / "task.py")], env=env, stdin=subprocess.DEVNULL,
                                 capture_output=True, text=True, timeout=10)
            self.assertNotEqual(out.returncode, 0)
            self.assertIn("set TASK=", out.stderr)


if __name__ == "__main__":
    unittest.main()
