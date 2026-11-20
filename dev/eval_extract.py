#!/usr/bin/env python3
"""eval_extract.py - how faithful is the extraction? (development only; needs `pip install --target dev/_deps pyyaml tomli` for parts B and C)

A. argparse defaults of 6 public PyTorch example scripts (dev/fixtures/public, BSD-3-Clause): every `add_argument('--name', ..., default=VALUE)` found by a regular expression over the
   source text (an independent route from the AST reader) must come out of extract.from_python with the same name and value.
B. mini YAML reader vs PyYAML on public YAML files of installed tools (leaf scalars compared after turning numeric-looking strings into numbers on both sides).
C. mini TOML reader vs tomli on public TOML files.
Usage: python3 dev/eval_extract.py [yaml_dir ...]
"""
import ast
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, next(p for p in (os.path.join(HERE, "..", "submission", "scripts"), os.path.join(HERE, "..", "scripts")) if os.path.isdir(p)))
sys.path.insert(0, os.path.join(HERE, "_deps"))
import extract  # noqa: E402
import mini  # noqa: E402


def part_a():
    tot = found = same = 0
    miss = []
    for p in sorted(glob.glob(os.path.join(HERE, "fixtures", "public", "*.py"))):
        src = open(p, encoding="utf-8").read()
        facts, note = extract.from_python(src, os.path.basename(p))
        got = {}
        for f in facts:
            if f["kind"] == "argparse":
                got[f["name"].replace("-", "_")] = f["value"]
        for m in re.finditer(r"add_argument\(\s*((?:'[^']*'\s*,\s*)*'--[\w-]+')(.*?)\)\s*(?:\n|$)", src, re.S):
            aliases = re.findall(r"'--([\w-]+)'", m.group(1))
            aliases = [a.replace("-", "_") for a in aliases]
            name, rest = next((a for a in aliases if a in got), aliases[0]), m.group(2)
            d = re.search(r"default\s*=\s*([^,\n)]+)", rest)
            if not d:
                continue
            try:
                want = ast.literal_eval(d.group(1).strip())
            except Exception:
                continue
            if want is None:
                continue  # a None default is not a fact
            tot += 1
            if name in got:
                found += 1
                g = got[name]
                if g == want or (isinstance(g, (int, float)) and isinstance(want, (int, float)) and abs(g - want) < 1e-12):
                    same += 1
                else:
                    miss.append((os.path.basename(p), name, want, g))
            else:
                miss.append((os.path.basename(p), name, want, "NOT FOUND"))
    print("A. argparse defaults in 6 public scripts: %d expected, %d found (%d%%), %d with the same value (%d%%)" % (tot, found, 100 * found // max(1, tot), same, 100 * same // max(1, tot)))
    for x in miss[:8]:
        print("   miss:", x)


def fix_keys(o):
    """PyYAML (YAML 1.1) reads the key `on` as True; the mini reader keeps `on`"""
    if isinstance(o, dict):
        return {("on" if k is True else ("off" if k is False else k)): fix_keys(v) for k, v in o.items()}
    if isinstance(o, list):
        return [fix_keys(x) for x in o]
    return o


def norm_leaves(o):
    out = {}
    for p, v in mini.flatten(fix_keys(o)):
        if isinstance(v, str):
            v = re.sub(r"\s+", " ", v).strip()
        if isinstance(v, str):
            n = extract.to_num(v)
            if n is not None and not v.strip().endswith("%"):
                v = n
        if isinstance(v, bool) or v is None:
            pass
        elif isinstance(v, int):
            v = float(v)
        if isinstance(v, list):
            v = [float(x) if isinstance(x, int) and not isinstance(x, bool) else x for x in v]
        out[p] = v
    return out


def part_b(dirs):
    try:
        import yaml
    except ImportError:
        print("B. skipped (no PyYAML)")
        return
    files = []
    for d in dirs:
        files += glob.glob(os.path.join(d, "**", "*.y*ml"), recursive=True)
    files = sorted(set(files))[:400]
    ok = same = oracle_fail = 0
    bad = []
    for p in files:
        try:
            text = open(p, encoding="utf-8").read()
            want = yaml.safe_load(text)
        except Exception:
            oracle_fail += 1
            continue
        if not isinstance(want, (dict, list)):
            continue
        try:
            got = mini.parse_yaml(text)
            ok += 1
        except Exception as e:
            bad.append((p, "exception %s" % type(e).__name__))
            continue
        a, b = norm_leaves(want), norm_leaves(got)
        if a == b:
            same += 1
        else:
            diff = [k for k in set(a) | set(b) if a.get(k) != b.get(k)][:2]
            bad.append((p, "leaf differs: %s" % diff))
    n = ok + len([b for b in bad if b[1].startswith("exception")])
    print("B. YAML: %d public files PyYAML can read (%d it cannot); mini reader: no exception on %d, identical scalar leaves on %d (%d%%)" % (n, oracle_fail, ok, same, 100 * same // max(1, n)))
    for x in bad[:6]:
        print("   ", x[1], "|", os.path.relpath(x[0], "/"))


def part_c():
    try:
        import tomli
    except ImportError:
        print("C. skipped (no tomli)")
        return
    files = sorted(glob.glob(os.path.join(HERE, "fixtures", "public_toml", "*")))
    same = tot = 0
    for p in files:
        try:
            text = open(p, encoding="utf-8").read()
            want = tomli.loads(text)
        except Exception:
            continue
        tot += 1
        try:
            got = mini.parse_toml(text)
        except Exception:
            continue
        if norm_leaves(want) == norm_leaves(got):
            same += 1
    print("C. TOML: %d public files, identical scalar leaves on %d" % (tot, same))


if __name__ == "__main__":
    part_a()
    part_b(sys.argv[1:] or [os.path.join(HERE, "fixtures", "public_yaml")])
    part_c()
