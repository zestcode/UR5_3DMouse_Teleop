"""Task choice at start-up (run by teleop.sh before the robot moves): e = continue an existing task, n = new task.

A task is the folder OUT_DIR/<task name>/: its episodes <NNN>_<label>/ are numbered from 001 within the task, and
_runs/<start time>/ holds session.json + robot.jsonl of each teleop run (written by recorder.py).
Questions go to stderr, the chosen task name to stdout. TASK=<name> skips the questions (tests, scripts). Ctrl-C
aborts. No folder is created here: the recorder creates a new task's folder when it starts."""
import os
from pathlib import Path
import re
import sys
import termios
import tty


def clean_name(text):
    """Folder name for a task: letters, digits, _ . - kept; spaces and other characters -> _."""
    return re.sub(r"[^\w.-]+", "_", text.strip()).strip("_.-")


def tasks(root):
    """-> [(name, episode count, last episode name or None)] of the task folders in root, most recently changed
    first. Folders starting with _ or . (e.g. _logs) are not tasks."""
    found = []
    for path in root.iterdir() if root.is_dir() else []:
        if path.is_dir() and not path.name.startswith(("_", ".")):
            episodes = sorted((int(m.group(1)), p.name) for p in path.iterdir()
                              if p.is_dir() and (m := re.match(r"(\d+)_", p.name)))
            found.append((path.stat().st_mtime, path.name, len(episodes), episodes[-1][1] if episodes else None))
    return [entry[1:] for entry in sorted(found, reverse=True)]


def say(text, end="\n"):
    sys.stderr.write(text + end)
    sys.stderr.flush()


def ask_key(prompt, keys):
    """One key press (no Enter) out of `keys`."""
    say(prompt, end="")
    fd = sys.stdin.fileno()
    attrs = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while (key := sys.stdin.read(1)) not in keys:
            if not key:
                raise EOFError
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
    say(key)
    return key


def ask_line(prompt):
    say(prompt, end="")
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    return line.strip()


def choose(root):
    found = tasks(root)
    if not found:
        say(f"No tasks in {root} yet: starting a new one.")
        key = "n"
    else:
        say(f"Tasks in {root}:")
        for i, (name, count, last) in enumerate(found, 1):
            say(f"  {i}  {name}  ({count} episode{'' if count == 1 else 's'}" + (f", last {last})" if last else ")"))
        key = ask_key("e = continue an existing task, n = new task: ", "eEnN").lower()
    if key == "e":
        if len(found) == 1:
            return found[0][0]
        if len(found) <= 9:  # one key press picks the task
            numbers = "".join(str(i) for i in range(1, len(found) + 1))
            return found[int(ask_key(f"Press the task number (1-{len(found)}): ", numbers)) - 1][0]
        while not ((text := ask_line(f"Task number (1-{len(found)}), then Enter: ")).isdigit()
                   and 1 <= int(text) <= len(found)):
            say("  not one of the task numbers")
        return found[int(text) - 1][0]
    while True:
        text = ask_line("Name of the new task: ")
        name = clean_name(text)
        if not name:
            say("  the name needs at least one letter or digit")
        elif (root / name).exists():
            say(f"  task {name} already exists: choose another name (or restart and press e to continue it)")
        else:
            if name != text:
                say(f"  folder name: {name}")
            return name


def main():
    root = Path(os.getenv("OUT_DIR") or Path(__file__).resolve().parent / "data")
    if os.getenv("TASK"):
        name = clean_name(os.environ["TASK"])
        if not name:
            sys.exit(f"✗ TASK={os.environ['TASK']!r} has no letter or digit")
    elif not sys.stdin.isatty():
        sys.exit("✗ No terminal to ask for the task: set TASK=<task name>")
    else:
        try:
            name = choose(root)
        except (KeyboardInterrupt, EOFError):
            say("\nCancelled")
            sys.exit(1)
    print(name)


if __name__ == "__main__":
    main()
