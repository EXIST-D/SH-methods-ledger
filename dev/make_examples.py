#!/usr/bin/env python3
"""make_examples.py - regenerate submission/examples/*/out and run.txt from real runs (nothing typed by hand). Everything in the examples is INVENTED."""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUB = os.path.join(ROOT, "submission") if os.path.isdir(os.path.join(ROOT, "submission")) else ROOT
EX = os.path.join(SUB, "examples")
MW = os.path.join(SUB, "scripts", "mw.py")
KEEP = {"project": ("section.md", "parameters.md", "questions.md", "source_map.csv", "review_checklist.md", "result.json", "brief.md", "commands.md"), "zh": ("section.md", "questions.md", "source_map.csv", "review_checklist.md")}


def scrub(text, *paths):
    for p in sorted(set(x for x in paths if x), key=len, reverse=True):
        text = text.replace(p, "<dir>")
    text = re.sub(r"/(?:Users|home|var/folders|private)/[^\s\"',)]*", "<dir>", text)
    return re.sub(r"<dir>/tmp\w+", "<work>", text)


def sh(cwd, args, log, cut=None):
    r = subprocess.run([sys.executable, MW] + args, capture_output=True, text=True, cwd=cwd)
    out = r.stdout + r.stderr
    if cut:
        lines = out.splitlines()
        out = "\n".join(lines[:cut] + (["... (%d more lines; the full brief is out/brief.md)" % (len(lines) - cut)] if len(lines) > cut else []))
    log.write(scrub("$ mw.py " + " ".join(args) + "\n" + out + "\n\n", EX, tempfile.gettempdir()))


def tidy(out, name):
    for f in os.listdir(out):
        p = os.path.join(out, f)
        if f not in KEEP[name]:
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)


def example(name, run_args, drafts, cut):
    d = EX if name == "project" else os.path.join(EX, name)
    out = os.path.join(d, "out")
    shutil.rmtree(out, ignore_errors=True)
    if os.path.isfile(os.path.join(d, "run.txt")) and name != "project":
        os.remove(os.path.join(d, "run.txt"))  # a previous transcript in the case folder would be read as a material
    work = tempfile.mkdtemp()
    log = open(os.path.join(tempfile.gettempdir(), "run_%s.txt" % name), "w", encoding="utf-8")
    sh(d, ["run"] + run_args + ["--out", "out", "--work", work], log, cut=cut)
    for kind, dr in drafts:
        sh(d, [kind, work, dr] + (["--out", "out"] if kind == "build" else []), log)
    log.close()
    tidy(out, name)
    shutil.move(os.path.join(tempfile.gettempdir(), "run_%s.txt" % name), os.path.join(d, "run.txt"))
    shutil.rmtree(work, ignore_errors=True)


def main():
    example("project", ["--src", "project"], [("check", "draft_v1.md"), ("check", "draft.md"), ("build", "draft.md")], 45)
    example("zh", ["--case", "case.json"], [("check", "draft.md"), ("build", "draft.md")], 30)


if __name__ == "__main__":
    main()
