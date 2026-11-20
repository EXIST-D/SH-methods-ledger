#!/usr/bin/env python3
"""mw.py - methods-ledger: write the Methods / Experiments section from the experiment's own files, and prove every number in it.

The facts (parameters, datasets, environment, commands, results) are read from configuration files, code, logs, result tables, requirement files and notes, each with the file and line it comes from.
Sources that disagree are listed as conflicts until the author decides. You write the section; `check` ties every number and parameter name in it to a fact (or marks it unsourced), catches values that
differ from the sources, training / evaluation values mixed up, missing elements and generic limitations; `build` writes the section, the parameter / dataset / environment tables, the questions for the author
and result.json.

  mw.py run [--case /input/case.json] [--src DIR ...] --out OUT        read everything, print the brief (facts, conflicts, missing elements, questions)
  mw.py show W [--facts PATTERN] [--conflicts] [--results]            look at the facts
  mw.py skeleton W [--out FILE]                                       a fact-bound outline with [[CONFIRM: ...]] where the materials are silent
  mw.py ingest W [FILE|-]                                             resolve / confirm blocks (author decisions, with evidence)
  mw.py check W DRAFT.md                                              ties numbers to sources, finds mismatches / mixups / gaps
  mw.py build W DRAFT.md --out OUT [--case ...] [--answers a.json]    section + tables + questions + result.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import docxw  # noqa: E402
import extract  # noqa: E402
import facts as F  # noqa: E402
import nlib  # noqa: E402

VERSION = "1.0.0"
BIN_EXT = (".pt", ".pth", ".ckpt", ".bin", ".npy", ".npz", ".pkl", ".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".h5", ".hdf5", ".zip", ".gz", ".tar", ".onnx", ".pdf_", ".mp4", ".wav", ".so", ".pyc", ".whl", ".safetensors", ".parquet", ".feather", ".nc", ".xlsx", ".pptx")
SKIP_NAMES = {"task.md", "case.json"}
DRAFT_HINT = re.compile(r"draft|section|manuscript|existing|current|初稿|章节|已有|现有|稿", re.I)


# ------------------------------------------------------------------------------------------------ io
def jload(p, default=None):
    if not os.path.isfile(p):
        return default
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def jsave(p, obj):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, p)


def wtext(p, text):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)


def rtext(p):
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def find_key(obj, names):
    if isinstance(obj, dict):
        for k in names:
            if k in obj:
                return obj[k]
        for v in obj.values():
            r = find_key(v, names)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, names)
            if r is not None:
                return r
    return None


def strings(obj, key=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from strings(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v, key)
    elif isinstance(obj, str):
        yield key, obj


def load(work):
    M = jload(os.path.join(work, "materials.json"))
    if M is None:
        raise SystemExit("no materials.json in %s: run `run` first" % work)
    return M, jload(os.path.join(work, "ops.json"), {"resolutions": [], "confirms": []})


# ------------------------------------------------------------------------------------------------ discovery
def walk(roots, depth=5, limit=700):
    out = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        base = root.rstrip(os.sep).count(os.sep)
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in sorted(dn) if d not in extract.SKIP_DIRS and not d.startswith(".") and d not in ("data", "datasets", "checkpoints", "ckpt", "wandb", "runs_cache")]
            if dp.count(os.sep) - base >= depth:
                dn[:] = []
            for f in sorted(fn):
                p = os.path.join(dp, f)
                if f.lower() in SKIP_NAMES or f.lower().endswith(BIN_EXT) or (f.startswith(".") and not f.startswith(".env")):
                    continue
                try:
                    if os.path.getsize(p) > 3_000_000:
                        continue
                except OSError:
                    continue
                if p not in out and len(out) < limit:
                    out.append(p)
    return out


def task_info(case, roots):
    text = ""
    for root in roots:
        p = os.path.join(root, "task.md")
        if os.path.isfile(p):
            text = rtext(p)
            break
    quotes = []
    if case:
        for key, val in strings(case):
            if re.search(r"task|instruction|prompt|requirement|任务|要求|说明|description", key, re.I) and len(val) > 15:
                text += "\n" + val
    info = {"text": text.strip()[:6000], "lang": None, "length": None, "venue": None}
    if re.search(r"中文|汉语|Chinese", text, re.I) and not re.search(r"English only|仅英文|用英文", text):
        info["lang"] = "zh"
    if re.search(r"用英文|英文撰写|in English|English version", text, re.I):
        info["lang"] = "en"
    m = re.search(r"(\d{3,5})\s*(?:字|个字|words|词)", text)
    if m:
        info["length"] = int(m.group(1))
    return info


def is_draft(path, text):
    base = os.path.basename(path)
    if extract.classify(path, text) != "doc" or not path.lower().endswith((".md", ".docx", ".tex", ".txt", ".markdown")):
        return False
    heads = re.findall(r"(?m)^#{1,4}\s+(.+)$", text) or [l for l in text.splitlines() if len(l) < 60 and re.search(r"方法|实验|Method|Experiment", l)]
    has_m = any(re.search(r"method|approach|model|方法|模型", h, re.I) for h in heads)
    has_e = any(re.search(r"experiment|setup|evaluation|implementation|实验|设置|评估", h, re.I) for h in heads)
    long_prose = len(re.findall(r"[.。]\s", text)) > 8
    return bool(DRAFT_HINT.search(base)) or (has_m and has_e and long_prose and not re.match(r"readme", base, re.I))


def prepare(a):
    os.makedirs(a.work, exist_ok=True)
    case = jload(a.case) if a.case and os.path.isfile(a.case) else None
    base = os.path.dirname(os.path.abspath(a.case)) if a.case else "/input"
    roots = list(dict.fromkeys([os.path.abspath(s) for s in (a.src or [])] + [base, "/input"]))
    for s in a.src or []:
        if not os.path.isdir(s):
            raise SystemExit("ERROR: source folder not found: %s" % s)
    paths = walk(roots)
    named = []
    if case:
        for key, val in strings(case):
            if len(val) < 300 and re.search(r"\.[A-Za-z0-9]{1,5}$", val.strip()):
                for root in roots + [os.getcwd()]:
                    p = val if os.path.isabs(val) else os.path.join(root, val)
                    if os.path.isfile(p) and os.path.abspath(p) not in paths and os.path.basename(p).lower() not in SKIP_NAMES:
                        paths.append(os.path.abspath(p))
                        named.append(os.path.abspath(p))
                        break
    for e in a.file or []:
        if not os.path.isfile(e):
            raise SystemExit("ERROR: file not found: %s" % e)
        paths.append(os.path.abspath(e))
    allfacts, commands, tables, files, drafts, notes = [], [], [], [], [], []
    common = os.path.commonpath([os.path.dirname(p) for p in paths]) if paths else ""
    for p in paths:
        rel = os.path.relpath(p, common) if common else os.path.basename(p)
        try:
            res = extract.extract_file(p, rel)
        except Exception as ex:  # one unreadable file must not stop the run
            files.append({"file": rel, "kind": "unreadable", "facts": 0, "note": "%s: %s" % (type(ex).__name__, str(ex)[:80])})
            continue
        text = res.get("text", "")
        if res["kind"] == "doc" and is_draft(p, text):
            drafts.append({"file": rel, "path": p})
            files.append({"file": rel, "kind": "draft (existing section: its numbers are checked, not used as evidence)", "facts": 0, "note": ""})
            continue
        allfacts += res["facts"]
        commands += res["commands"]
        tables += res["tables"]
        files.append({"file": rel, "kind": res["kind"], "facts": len(res["facts"]), "note": res["note"]})
    quote_pool = []
    if case:
        for key, val in strings(case):
            if len(val) > 120 and not re.search(r"output_contract", key):
                notes.append({"key": key, "len": len(val)})
                fs = extract.from_prose(val, "case.json#%s" % key, "doc")
                allfacts += fs
                fc, cm = extract.from_commands(val, "case.json#%s" % key, "doc")
                allfacts += fc
                commands += cm
                quote_pool.append(val)
        for key, sub in (find_key_items(case)):
            fs, note = extract.from_config("x.json", json.dumps(sub), "case.json#%s" % key)
            for f in fs:
                f["kind"] = "config"
            allfacts += fs
    task = task_info(case, roots)
    if task["text"]:
        quote_pool.append(task["text"])
        tf = extract.from_prose(task["text"], "task.md", "doc")
        allfacts += tf
    extract.assign_ids(allfacts)
    # a text file of the author's own words is also a quote pool (to confirm facts the files do not hold)
    for p in paths:
        if extract.classify(p, "") == "doc" or p.lower().endswith((".md", ".txt", ".docx")):
            try:
                quote_pool.append(extract.read_text(p)[0])
            except Exception:
                pass
    M = {"version": VERSION, "facts": allfacts, "commands": commands, "tables": tables, "files": files, "drafts": drafts, "task": task, "case_id": (find_key(case, ["case_id", "caseId"]) or find_key(case, ["id"])) if case else None,
         "case": case, "quote_pool": "\n\n".join(quote_pool)[:400000], "lang": task["lang"] or nlib.detect_lang(" ".join(f["context"] for f in allfacts if f["kind"] == "doc")[:3000] or "en")}
    jsave(os.path.join(a.work, "materials.json"), M)
    if not os.path.isfile(os.path.join(a.work, "ops.json")):
        jsave(os.path.join(a.work, "ops.json"), {"resolutions": [], "confirms": []})
    return M


def find_key_items(case):
    """dict sub-trees of the case whose key says they are settings (config / params / hyper-parameters / 参数 ...)"""
    out = []

    def walk_(o, k=""):
        if isinstance(o, dict):
            for kk, v in o.items():
                if isinstance(v, dict) and re.search(r"config|param|hyper|setting|setup|args|参数|配置|设置", kk, re.I):
                    out.append((kk, v))
                else:
                    walk_(v, kk)
        elif isinstance(o, list):
            for v in o:
                walk_(v, k)
    walk_(case)
    return out


# ------------------------------------------------------------------------------------------------ derived structures
def analysis(M, ops):
    fs = M["facts"]
    confl = F.conflicts(fs)
    res = ops["resolutions"]
    table = F.param_table(fs, confl, res)
    return {"conflicts": confl, "table": table, "datasets": F.dataset_rows(fs), "env": F.env_rows(fs), "results": F.results_rows(fs)}


def open_conflicts(confl, res):
    return [c for c in confl if not F.resolution_for(res, c["canon"], c["scope"])]


def brief(work, out, case_note=True):
    M, ops = load(work)
    A = analysis(M, ops)
    fs = M["facts"]
    els = F.elements(fs, M["commands"], "")
    L = ["# Evidence brief (generated; every value below is written in a file, with its source)\n"]
    t = M["task"]
    L.append("## Task\n- language: %s | length hint: %s\n%s\n" % (M["lang"], t["length"] or "none", ("```\n" + t["text"][:1200] + "\n```") if t["text"] else "- no task text found"))
    L.append("## Files read (%d)\n" % len(M["files"]) + "\n".join("- %s [%s]: %d facts%s" % (f["file"], f["kind"], f["facts"], (" - " + f["note"]) if f["note"] else "") for f in M["files"][:60]))
    L.append("\n## Parameters (resolved value, scope, status, sources)")
    for r in A["table"][:90]:
        L.append("- %s = %s [%s] %s | %s" % (r["name"], F.show(r["value"]) if not isinstance(r["value"], list) else r["value"], r["scope"], r["status"], "; ".join(r["sources"][:3])))
    if len(A["table"]) > 90:
        L.append("... %d more (show W --facts)" % (len(A["table"]) - 90))
    L.append("\n## Conflicts between sources (%d open; decide with `=== resolve`)" % len(open_conflicts(A["conflicts"], ops["resolutions"])))
    for c in A["conflicts"]:
        done = F.resolution_for(ops["resolutions"], c["canon"], c["scope"])
        L.append("- %s [%s]%s: %s | %s" % (c["canon"], c["scope"], " (resolved)" if done else "", "  vs  ".join("%s @ %s" % (F.show(v["value"]), ", ".join(v["sources"][:2])) for v in c["values"]), c["hint"]))
    L.append("\n## Datasets and environment")
    for r in A["datasets"][:25]:
        L.append("- %s = %s | %s" % (r["item"], F.show(r["value"]) if not isinstance(r["value"], list) else r["value"], r["source"]))
    for r in A["env"][:25]:
        L.append("- %s = %s | %s" % (r["item"], r["value"], r["source"]))
    L.append("\n## Results found (last value per metric)")
    for name, fl in list(A["results"].items())[:30]:
        L.append("- %s = %s | %s%s" % (name, F.show(fl[-1]["value"]), fl[-1]["source"], " (%d values)" % len(fl) if len(fl) > 1 else ""))
    L.append("\n## Commands found (%d)" % len(M["commands"]))
    for c in M["commands"][:12]:
        L.append("- %s:%s `%s`" % (c["file"], c["line"], c["command"][:140]))
    miss = [e for e in els if e["status"] == "unknown_from_material"]
    L.append("\n## Not in the materials (%d): ask the author, never fill in" % len(miss))
    for e in miss:
        L.append("- %s: %s" % (e["id"], F.QUESTION[e["id"]]))
    if M["drafts"]:
        L.append("\n## Existing draft(s): %s (checked below; revise it rather than start over)" % ", ".join(d["file"] for d in M["drafts"]))
    txt = "\n".join(L) + "\n"
    wtext(os.path.join(out, "brief.md"), txt)
    print(txt)
    print("[brief written to %s]" % os.path.join(out, "brief.md"))
    return M, A


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    a.work = a.work or os.path.join(a.out, ".work")
    M = prepare(a)
    brief(a.work, a.out)
    if M["drafts"]:
        print("\n--- check of the existing draft %s" % M["drafts"][0]["file"])
        ns = argparse.Namespace(work=a.work, draft=M["drafts"][0]["path"], all=False)
        cmd_check(ns)
    print("Next: write the section (Markdown) from these facts, `check` it, answer the open conflicts with `ingest` where the task text allows, then `build`.")


def cmd_show(a):
    M, ops = load(a.work)
    A = analysis(M, ops)
    if a.conflicts:
        for c in A["conflicts"]:
            print("%s [%s]: %s | %s" % (c["canon"], c["scope"], "  vs  ".join("%s @ %s" % (F.show(v["value"]), ", ".join(v["sources"])) for v in c["values"]), c["hint"]))
        return
    if a.results:
        for name, fl in A["results"].items():
            print("%s: %s" % (name, ", ".join("%s@%s" % (F.show(f["value"]), f["source"]) for f in fl[:12])))
        return
    rx = re.compile(a.facts or ".", re.I)
    n = 0
    for f in M["facts"]:
        if rx.search("%s %s %s %s" % (f["name"], f["canon"], f["value"], f["file"])):
            print("%s %-8s %-28s = %s [%s] %s%s" % (f["id"], f["kind"], f["name"][:28], f["value"] if not isinstance(f["value"], list) else f["value"], f["scope"], f["source"], ("  // " + f["context"][:70]) if f["context"] else ""))
            n += 1
            if n >= a.limit:
                print("... (limit %d)" % a.limit)
                break


# ------------------------------------------------------------------------------------------------ ingest
def parse_blocks(text):
    recs, cur, key = [], None, None
    for ln in text.splitlines():
        s = ln.rstrip()
        if re.match(r"^\s*```\w*\s*$", s):
            continue
        m = re.match(r"^\s*[#=*]{2,}\s*(resolve|confirm|limitation)\b\s*(.*?)[*=#]*\s*$", s, re.I)
        if m:
            cur = {"kind": m.group(1).lower(), "target": m.group(2).strip(), "f": {}}
            recs.append(cur)
            key = None
            continue
        if cur is None:
            continue
        mk = re.match(r"^\s*(?:[-*]\s*)?([A-Za-z_]+)\s*[:：]\s*(.*)$", s)
        if mk and mk.group(1).lower() in ("value", "scope", "reason", "from", "unit", "quote", "note", "scenario", "evidence", "name"):
            key = mk.group(1).lower()
            cur["f"][key] = mk.group(2).strip()
        elif key and s.strip():
            cur["f"][key] += " " + s.strip()
    return recs


def cmd_ingest(a):
    M, ops = load(a.work)
    text = sys.stdin.read() if (not a.file or a.file == "-") else rtext(a.file)
    blocks = parse_blocks(text)
    if not blocks:
        raise SystemExit("no blocks found: '=== resolve NAME', '=== confirm NAME', '=== limitation L1' followed by key: value lines")
    A = analysis(M, {"resolutions": [], "confirms": []})
    n, msgs, bad = Counter(), [], 0
    for b in blocks:
        k, f, tgt = b["kind"], b["f"], b["target"]
        if k == "resolve":
            name = tgt.split()[0] if tgt else f.get("name", "")
            canon = extract.canon_of(name) or name
            scope = f.get("scope") or (tgt.split()[1] if len(tgt.split()) > 1 else "any")
            cl = [c for c in A["conflicts"] if c["canon"] == canon and scope in (c["scope"], "any")]
            if not cl:
                msgs.append("resolve %s: no open conflict for it (conflicts: %s)" % (canon, ", ".join("%s[%s]" % (c["canon"], c["scope"]) for c in A["conflicts"]) or "none"))
                bad += 1
                continue
            val = f.get("value", "")
            seen = {F.vkey(v["value"]) for c in cl for v in c["values"]}
            chosen = next((v["value"] for c in cl for v in c["values"] if F.vkey(v["value"]) == F.vkey(mini_scalar(val))), None)
            if chosen is None:
                msgs.append("resolve %s REJECTED: value '%s' is none of the observed values (%s); an author-confirmed new value needs `=== confirm` with a quote from the author's own words" % (canon, val, ", ".join(sorted(F.show(v["value"]) for c in cl for v in c["values"]))))
                bad += 1
                continue
            reason = f.get("reason", "")
            if not reason:
                msgs.append("resolve %s REJECTED: give a reason (which file is the run that produced the results, or the author's statement)" % canon)
                bad += 1
                continue
            ops["resolutions"] = [r for r in ops["resolutions"] if not (r["canon"] == canon and r["scope"] == scope)]
            ops["resolutions"].append({"canon": canon, "scope": scope if scope else "any", "value": chosen, "reason": reason, "source": f.get("from", "")})
            n["resolve"] += 1
        elif k == "confirm":
            name = tgt or f.get("name", "")
            quote = f.get("quote", "").strip(" \"'“”")
            val = mini_scalar(f.get("value", ""))
            pool = M["quote_pool"]
            if not quote or nlib.squash(quote) not in nlib.squash(pool) or len(nlib.squash(quote)) < 8:
                msgs.append("confirm %s REJECTED: the quote is not in the task text / notes / documents given with the case. A value nobody wrote down goes to the questions for the author, not into the ledger" % name)
                bad += 1
                continue
            if val is not None and not isinstance(val, str) and not any(F.vkey(val) == F.vkey(x) for x in numbers_in(quote)):
                msgs.append("confirm %s REJECTED: value %s is not in the quote" % (name, val))
                bad += 1
                continue
            ops["confirms"] = [c for c in ops["confirms"] if c["name"] != name]
            ops["confirms"].append({"name": name, "value": val, "unit": f.get("unit", ""), "scope": f.get("scope", "general"), "quote": quote, "note": f.get("note", "")})
            n["confirm"] += 1
        elif k == "limitation":
            ops.setdefault("limitations", [])
            sc = f.get("scenario", "")
            ev = f.get("evidence", "")
            if len(sc.split()) < 6 and len(sc) < 12:
                msgs.append("limitation %s REJECTED: describe the failure scenario (when it fails, for what input or setting)" % tgt)
                bad += 1
                continue
            ops["limitations"] = [x for x in ops["limitations"] if x["id"] != tgt]
            ops["limitations"].append({"id": tgt, "scenario": sc, "evidence": ev})
            n["limitation"] += 1
    jsave(os.path.join(a.work, "ops.json"), ops)
    print("recorded: " + ", ".join("%s %d" % kv for kv in n.items()) + (" | rejected %d" % bad if bad else ""))
    for m in msgs[:20]:
        print("  " + m)
    sys.exit(2 if bad else 0)


def mini_scalar(v):
    import mini
    return mini.scalar(str(v)) if v != "" else None


def numbers_in(text):
    out = []
    for m in NUMTOK.finditer(text):
        out.append(parse_tok(m)[0])
    return out


# ------------------------------------------------------------------------------------------------ draft parsing and number matching
NUMTOK = re.compile(r"(?<![A-Za-z0-9_.\-/\\])(?P<neg>[-−–]?)(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:\s*[×x]\s*10\s*\^?\s*\{?(?P<exp>[-−–]?\d+)\}?|[eE](?P<e2>[-+]?\d+)(?![A-Za-z]))?(?P<suf>\s?%|\s?[kK](?![A-Za-z])|\s?[mM](?![A-Za-z]))?")
VERSION_TOK = re.compile(r"(?<![\w.])v?\d+(?:\.\d+){2,}(?:[+\-]\w+)?(?![\w.]*\d)")
CLEAN = [re.compile(p, re.I) for p in (
    r"\[@[^\]]*\]", r"\[\d+(?:\s*[,–-]\s*\d+)*\]", r"\([A-Z][A-Za-z'\-]+(?: et al\.?)?(?: and [A-Z][a-z]+)?,? (?:19|20)\d{2}[a-z]?\)", r"https?://\S+", r"\b(?:Tables?|Figures?|Figs?\.?|Sections?|Sec\.?|Eqs?\.?|Equations?|Algorithms?|Steps?|Stages?|Phases?|Appendix|Lines?)\s*\(?[A-Z]?\d+(?:\.\d+)*\)?",
    r"(?:图|表|第|步骤|阶段|式|公式|附录)\s*\d+(?:\.\d+)*", r"^\s*\d+(?:\.\d+)*[.)、]\s+", r"(?m)^#+\s+\d+(?:\.\d+)*", r"`[^`]*`", r"\[\[CONFIRM:[^\]]*\]\]")]
RATIO = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*(?:%\s*)?[:/：]\s*(\d{1,3}(?:\.\d+)?)\s*(?:%\s*)?[:/：]\s*(\d{1,3}(?:\.\d+)?)\s*%?(?![\d.])")


def parse_tok(m):
    v = float(m.group("num").replace(",", ""))
    if m.group("neg"):
        v = -v
    if m.group("exp"):
        v = v * (10 ** int(m.group("exp").replace("−", "-").replace("–", "-")))
    elif m.group("e2"):
        v = v * (10 ** int(m.group("e2")))
    suf = (m.group("suf") or "").strip().lower()
    alts = [v]
    if suf == "%":
        alts.append(v / 100.0)
    elif suf == "k":
        alts.append(v * 1e3)
    elif suf == "m":
        alts.append(v * 1e6)
    return alts[0], alts, suf


def close(a, b):
    return abs(a - b) <= max(1e-9, 1e-6 * max(abs(a), abs(b)))


def fnums(f):
    """numeric values of a fact (list members too)"""
    v = f["value"]
    out = []
    if isinstance(v, bool):
        return out
    if isinstance(v, (int, float)):
        out.append(float(v))
    elif isinstance(v, list):
        out += [float(x) for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
    elif isinstance(v, str):
        n = extract.to_num(v)
        if n is not None:
            out.append(n)
        for x in re.findall(r"\d+(?:\.\d+)?", v) if re.fullmatch(r"[\d.:/ %]+", v) else []:
            out.append(float(x))
    return out


def derived_numbers(facts):
    """numbers a reader can compute from dataset sizes: totals and shares"""
    sizes = {f["name"].split(".")[-1]: f["num"] for f in facts if f["name"].startswith("dataset.size.") and f["num"]}
    out = set()
    st = [sizes[k] for k in ("train", "val", "test") if k in sizes]
    if len(st) >= 2:
        tot = sum(st)
        out.add(tot)
        for x in st:
            out.add(round(x / tot, 4))
            out.add(round(100 * x / tot, 2))
            out.add(round(100 * x / tot, 1))
            out.add(round(100 * x / tot))
    if len(st) == 3:
        out.add(sizes["train"] + sizes["val"])
    return out


STAGE_CUES = (("train", re.compile(r"\b(train(?:ing|ed)?|fine-?tun\w*|pre-?train\w*|optimi[sz]\w*|learn\w*)\b|训练|微调", re.I)),
              ("val", re.compile(r"\b(validat\w+|val(?:id)?\b|dev(?:elopment)? set|model selection|early stopping)\b|验证", re.I)),
              ("test", re.compile(r"\b(test(?:ing|ed)?|evaluat\w+|inference|held-?out|benchmark(?:ed)?)\b|测试|评测|评估|推理", re.I)))


def stage_of(ctx):
    best, pos = None, -1
    for st, rx in STAGE_CUES:
        for m in rx.finditer(ctx):
            if m.start() >= pos:
                best, pos = st, m.start()
    return best


def canon_near(before, after):
    """the canonical parameter a number belongs to: first the noun that follows it (50 epochs, 4 GPUs, 500 warm-up steps), else the name just before it (learning rate 0.001)"""
    am = after[:40]
    wb = re.search(r"warm-?up|预热", before[-30:], re.I)
    for k in ("epochs", "num_gpus", "repeats", "warmup", "steps", "num_layers", "num_heads", "num_workers", "batch_size", "seed"):
        if k in ("steps", "epochs") and wb:
            continue
        gap = r"(?:x\s*)?(?:[A-Za-z0-9\-]+\s+){0,3}" if k == "num_gpus" else r"(?:training |independent |random |attention |hidden |transformer |linear )?"
        mm = re.match(r"\s*(?:个|次|张|块|层|轮|步)?\s*" + gap + r"(?:%s)" % extract.PARAMS[k][1], am, re.I)
        if mm:
            return k
    best, bd = None, 10 ** 9
    tail = before[-48:]
    for k, rx in extract.PHRASE_RX.items():
        for mm in rx.finditer(tail):
            gap = tail[mm.end():]
            if len(gap) <= 24 and not re.search(r"\d", gap) and not re.search(r"[.;。；]", gap):
                if len(gap) < bd:
                    best, bd = k, len(gap)
    return best


COMMON = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 0.1, 0.2, 0.3, 0.5, 0.25, 0.01, 0.001, 0.0001, 0.9, 0.99, 0.999, 100.0, 0.05, 0.02}


def split_sentences(text):
    out = []
    for para in re.split(r"\n\s*\n", text):
        p = re.sub(r"\s*\n\s*", " ", para).strip()
        if not p or re.match(r"^#{1,6}\s", p):
            continue
        out += nlib.split_sentences(p)
    return out


def parse_draft(text):
    """-> [{id, part, text, heading}] sentences with the part (heading) they sit in"""
    sents, part, sid = [], "", 0
    for block in re.split(r"\n\s*\n", text.strip()):
        b = block.strip()
        mh = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", b)
        if mh and "\n" not in b:
            part = mh.group(2).strip()
            continue
        if b.startswith(("```", "|", "---")) or re.match(r"^\s*(?:Table|Figure|表|图)\s*\d+", b):
            continue
        for line in b.split("\n"):
            line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line).strip()
            if not line:
                continue
            for s in nlib.split_sentences(line) or [line]:
                sid += 1
                sents.append({"id": "D%03d" % sid, "part": part, "text": s})
    return sents


def part_role(part):
    p = part.lower()
    if re.search(r"limit|局限|限制|threat|caveat", p):
        return "limitations"
    if re.search(r"reproduc|复现|availability|code|可复现|可用性", p):
        return "repro"
    if re.search(r"dataset|data\b|数据", p):
        return "dataset"
    if re.search(r"baseline|compar|对照|对比|基线", p):
        return "baseline"
    if re.search(r"setup|setting|implement|detail|training|evaluation|experiment|实验|设置|实现|训练|评估|评测", p):
        return "setup"
    if re.search(r"method|approach|model|algorithm|pipeline|方法|模型|算法|流程", p):
        return "method"
    return ""


GENERIC_LIMIT = re.compile(r"may not generali[sz]e|might not generali[sz]e|further (?:research|work|study|studies|investigation|experiments)|future (?:work|research|studies)|more (?:experiments|data|datasets) (?:are|is) needed|limited (?:by )?(?:data|samples?|scope|resources|time)|has some limitations|there are (?:several )?limitations|"
                         r"仍需|有待|进一步(?:研究|探索|验证)|可能不适用|泛化能力有限|存在一定局限|数据有限|未来(?:工作|研究)", re.I)
SCENARIO = re.compile(r"\d|\bwhen\b|\bif\b|\bunder\b|\bbeyond\b|\babove\b|\bbelow\b|\bfor (?:example|instance)\b|\be\.g\.|\bsuch as\b|\bonly (?:on|for|in)\b|\bnot (?:tested|evaluated|validated) (?:on|for|in|with)\b|\bfails?\b|\bdegrad\w+|"
                      r"当.{1,30}时|在.{1,30}(?:下|时|中)|超过|低于|高于|仅在|仅限|未在.{1,20}(?:上|中)(?:测试|验证|评估)|退化|失效")
COMPARABLE = re.compile(r"same (?:training |compute |computational |hyper-?parameter |tuning )?budget|fair(?:ly)? compar\w+|identical (?:settings|conditions|hyper-?parameters)|under the same (?:settings|conditions|protocol)|equal (?:budget|compute)|同等(?:预算|算力|条件)|相同(?:预算|条件|设置|超参数)|公平(?:比较|对比)", re.I)
PLACEHOLDER = re.compile(r"⟦[^⟧]*⟧|\bTODO\b|\bTBD\b|\bX{2,}\b|\[\?+\]|\?\?\?|待补(?:充)?|待填|\bN/A\b(?! \()", re.I)
CONFIRM = re.compile(r"\[\[CONFIRM:\s*([^\]]+)\]\]")
SEQ = re.compile(r"\b(?:then|next|after that|afterwards|finally|subsequently)\b|然后|接着|之后|最后|随后", re.I)


def issue(out, code, sev, msg, sid=None, **kw):
    out.append({"code": code, "severity": sev, "sentence": sid, "message": msg, **kw})


def check_draft(M, ops, text, A=None):
    fs = M["facts"]
    A = A or analysis(M, ops)
    res = ops["resolutions"]
    conf_facts = []
    for c in ops.get("confirms", []):
        v = c["value"]
        conf_facts.append({"id": "C:" + c["name"], "name": c["name"], "canon": extract.canon_of(c["name"]), "value": v, "num": extract.to_num(v) if v is not None else None, "scope": c.get("scope", "general"), "kind": "confirm", "file": "author", "line": None, "source": "author confirmation: \"%s\"" % c["quote"][:60], "context": c["quote"]})
    allf = fs + conf_facts
    sents = parse_draft(text)
    iss, rows = [], []
    derived = derived_numbers(allf)
    body = text
    # ---- numbers
    by_canon = defaultdict(list)
    for f in allf:
        if f["canon"]:
            by_canon[f["canon"]].append(f)
    for s in sents:
        raw = s["text"]
        clean = raw
        for rx in CLEAN:
            clean = rx.sub(lambda m: " " * len(m.group(0)), clean)
        clean = VERSION_TOK.sub(lambda m: " " * len(m.group(0)), clean)
        for rxv in (extract.PKGVER, extract.JAXVER):
            clean = rxv.sub(lambda m: " " * len(m.group(0)), clean)
        role = part_role(s["part"])
        spans = []
        for m in RATIO.finditer(clean):
            spans.append((m.start(), m.end()))
            parts = [float(x) for x in m.groups()]
            tot = sum(parts)
            prop = [round(x / tot, 4) for x in parts]
            ok, src = False, []
            for f in allf:
                if (f["name"] == "dataset.split_ratio" or f["canon"] == "test_size" or re.search(r"split", f["name"])) and isinstance(f["value"], (list, str)):
                    vals = [float(x) for x in (f["value"] if isinstance(f["value"], list) else re.findall(r"\d+(?:\.\d+)?", str(f["value"])))]
                    if len(vals) == 3 and sum(vals) > 0 and [round(x / sum(vals), 4) for x in vals] == prop:
                        ok, src = True, [f["source"]]
                        break
            if not ok and derived and any(close(prop[0], d) for d in derived):
                ok, src = True, ["derived from split sizes"]
            rows.append({"sentence": s["id"], "token": m.group(0), "status": "sourced" if ok else "unsourced", "sources": src, "kind": "split ratio"})
            if not ok:
                issue(iss, "unsourced_number", "error", "split ratio %s is not given by any source (dataset split, sizes)" % m.group(0), s["id"], token=m.group(0))
        for m in NUMTOK.finditer(clean):
            if any(a <= m.start() < b for a, b in spans):
                continue
            v, alts, suf = parse_tok(m)
            before, after = clean[max(0, m.start() - 70):m.start()], clean[m.end():m.end() + 40]
            canon = canon_near(before, after)
            ctx = clean[max(0, m.start() - 90):m.end() + 60]
            stage = stage_of(ctx)
            tokraw = m.group(0).strip()
            is_year = suf == "" and re.fullmatch(r"(?:19|20)\d{2}", m.group("num")) is not None
            cands = [f for f in allf if any(close(x, y) for x in alts for y in fnums(f))]
            if canon:
                cc = [f for f in cands if f["canon"] == canon]
                if not cc and canon == "test_size":
                    cc = [f for f in cands if f["canon"] == "test_size" or re.search(r"split|dataset\.size", f["name"])]
                known = by_canon.get(canon, [])
                if cc:
                    pick = pick_source(cc, stage)
                    mix = None
                    if stage in F.STAGES:
                        same = [f for f in known if f["scope"] == stage]
                        if same and not any(f in cc for f in same):
                            other = [f for f in cc if f["scope"] in F.STAGES and f["scope"] != stage]
                            if other:
                                mix = (stage, F.show(same[0]["value"]), same[0]["source"], other[0]["scope"])
                    if mix:
                        issue(iss, "condition_mixup", "error", "%s %s is the %s value (%s) but the sentence describes %s, where the sources say %s (%s)" % (canon.replace("_", " "), tokraw, mix[3], pick["source"], mix[0], mix[1], mix[2]), s["id"], token=tokraw)
                        rows.append({"sentence": s["id"], "token": tokraw, "status": "condition_mixup", "sources": [pick["source"]], "kind": canon})
                    else:
                        conf = [c for c in A["conflicts"] if c["canon"] == canon and not F.resolution_for(res, c["canon"], c["scope"]) and (c["scope"] == (stage or "general") or c["scope"] == "general")]
                        if conf and not CONFIRM.search(raw):
                            issue(iss, "conflicted_value_used", "warn", "%s %s is used while the sources disagree (%s): the author must settle it" % (canon.replace("_", " "), tokraw, "  vs  ".join("%s @ %s" % (F.show(v["value"]), v["sources"][0]) for v in conf[0]["values"])), s["id"], token=tokraw)
                        rows.append({"sentence": s["id"], "token": tokraw, "status": "sourced" + (" (conflicted)" if conf else ""), "sources": [f["source"] for f in cc][:3], "kind": canon})
                elif known:
                    vals = sorted({(F.show(f["value"]) if not isinstance(f["value"], list) else str(f["value"])) + " @ " + f["source"] for f in known})
                    issue(iss, "value_mismatch", "error", "the text says %s %s but the sources say %s" % (canon.replace("_", " "), tokraw, "; ".join(vals[:4])), s["id"], token=tokraw)
                    rows.append({"sentence": s["id"], "token": tokraw, "status": "value_mismatch", "sources": [f["source"] for f in known][:3], "kind": canon})
                else:
                    if any(close(x, d) for x in alts for d in derived) and False:
                        pass
                    issue(iss, "unsourced_number", "error", "%s %s: no file, log or note gives a %s" % (canon.replace("_", " "), tokraw, canon.replace("_", " ")), s["id"], token=tokraw)
                    rows.append({"sentence": s["id"], "token": tokraw, "status": "unsourced", "sources": [], "kind": canon})
                continue
            if cands and not (alts[0] in COMMON and not suf):
                pick = pick_source(cands, stage)
                rows.append({"sentence": s["id"], "token": tokraw, "status": "sourced", "sources": [f["source"] for f in cands][:3], "kind": pick["name"]})
                continue
            if cands:
                # a common value (1, 0.1, 0.5 ...) equals some fact by coincidence all the time: accept it only when the sentence shares a word with the fact's name or context
                import evidence_words
                sw = evidence_words.words(clean)
                near = [f for f in cands if sw & evidence_words.words(f["name"] + " " + f["context"])]
                if near:
                    pick = pick_source(near, stage)
                    rows.append({"sentence": s["id"], "token": tokraw, "status": "sourced", "sources": [f["source"] for f in near][:3], "kind": pick["name"]})
                    continue
            splitctx = re.search(r"split|ratio|proportion|fraction|share|percent|划分|比例|占比", ctx, re.I)
            if any(close(x, d) for x in alts for d in derived) and (not (alts[0] in COMMON and not suf) or splitctx):
                rows.append({"sentence": s["id"], "token": tokraw, "status": "derived", "sources": ["computed from dataset sizes"], "kind": "derived"})
                continue
            if is_year:
                rows.append({"sentence": s["id"], "token": tokraw, "status": "year (not checked)", "sources": [], "kind": "year"})
                continue
            if float(v).is_integer() and abs(v) <= 12 and suf == "" and "." not in m.group("num"):
                rows.append({"sentence": s["id"], "token": tokraw, "status": "small count (not checked)", "sources": [], "kind": "count"})
                continue
            issue(iss, "unsourced_number", "error", "%s is not found in any source (config, code, log, table, note): it may have been added for the sake of the prose%s" % (tokraw, " - context: \"%s\"" % ctx.strip()[:80]), s["id"], token=tokraw)
            rows.append({"sentence": s["id"], "token": tokraw, "status": "unsourced", "sources": [], "kind": ""})
        # ---- named values
        opt_known = {re.sub(r"[\s_\-]", "", str(f["value"]).lower()) for f in allf if f["canon"] == "optimizer"}
        for om in extract.OPT_RX.finditer(raw):
            nm = re.sub(r"[\s_\-]", "", om.group(1).lower())
            if opt_known and nm not in opt_known:
                issue(iss, "value_mismatch", "error", "the text says the optimizer is %s but the sources say %s" % (om.group(1), ", ".join(sorted({"%s @ %s" % (f["value"], f["source"]) for f in allf if f["canon"] == "optimizer"})[:3])), s["id"])
            elif not opt_known and re.search(r"optimi[sz]er|trained with|优化器", raw, re.I):
                issue(iss, "unsourced_name", "error", "optimizer %s is not named by any source" % om.group(1), s["id"])
        pk_known = defaultdict(set)
        for f in allf:
            if f["name"].startswith("env."):
                pk_known[canon_pkg(f["name"][4:])].add(str(f["value"]).lstrip(">=~"))
        seen_pk = set()
        for pm in list(extract.PKGVER.finditer(raw)) + list(extract.JAXVER.finditer(raw)):
            pk, ver = canon_pkg(pm.group(1)), pm.group(2).rstrip(".")
            if (pk, ver) in seen_pk:
                continue
            seen_pk.add((pk, ver))
            if pk in pk_known:
                if not any(ver == k or k.startswith(ver + ".") or ver.startswith(k + ".") for k in pk_known[pk]):
                    issue(iss, "value_mismatch", "error", "the text says %s %s but the sources say %s" % (pm.group(1), ver, ", ".join(sorted(pk_known[pk]))), s["id"])
            else:
                issue(iss, "unsourced_version", "error", "%s %s: no requirement file, environment file or log gives this version" % (pm.group(1), ver), s["id"])
        gpu_known = {re.sub(r"[^a-z0-9]", "", str(f["value"]).lower()) for f in allf if f["name"] == "hardware.gpu"}
        for gm in extract.GPU.finditer(raw):
            g = re.sub(r"[^a-z0-9]", "", gm.group(1).lower().replace("nvidia", ""))
            core = re.search(r"(a100|a800|a40|a30|a10g?|h100|h800|v100|p100|t4|l40s?|rtx\w*|gtx\w*|titan\w*|4090|3090|2080ti)", g)
            if gpu_known:
                if core and not any(core.group(1) in k for k in gpu_known):
                    issue(iss, "value_mismatch", "error", "the text says the GPU is %s but the sources say %s" % (gm.group(1), ", ".join(sorted({str(f["value"]) + " @ " + f["source"] for f in allf if f["name"] == "hardware.gpu"})[:3])), s["id"])
            else:
                issue(iss, "unsourced_name", "error", "GPU %s is not named by any source" % gm.group(1), s["id"])
        dv = {str(f["value"]).lower() for f in allf if re.search(r"version$", f["name"], re.I) and f["scope"] != "env" and f["kind"] != "env"}
        for vm in re.finditer(r"(?<![\w.])v(\d+(?:\.\d+)+)(?![\w.]*\d)|(?:version|版本)\s*(\d+(?:\.\d+)+)", raw, re.I):
            ver = vm.group(1) or vm.group(2)
            if re.search(r"\b(?:torch|pytorch|tensorflow|cuda|python)\b", raw[max(0, vm.start() - 25):vm.start()], re.I):
                continue
            if dv and not any(ver == x.lstrip("v") for x in dv):
                issue(iss, "value_mismatch", "error", "the text says dataset version %s but the sources say %s" % (ver, ", ".join(sorted(dv))), s["id"])
            elif not dv:
                issue(iss, "unsourced_version", "warn", "version %s is not found in any source" % ver, s["id"])
        if COMPARABLE.search(raw) and not any(c["name"].lower().startswith(("budget", "baseline")) for c in ops.get("confirms", [])):
            issue(iss, "unverified_comparability", "warn", "claims the comparison is fair / same budget: no source documents the baselines' budget; confirm with the author (`=== confirm baseline_budget` with a quote) or soften it", s["id"])
        if PLACEHOLDER.search(raw):
            issue(iss, "placeholder_left", "error", "unfinished placeholder in the text (%s): fill it from a source or mark it [[CONFIRM: what is needed]]" % PLACEHOLDER.search(raw).group(0)[:20], s["id"])
        if len(SEQ.findall(raw)) >= 3 and len(raw.split()) > 40:
            issue(iss, "compressed_steps", "info", "one sentence compresses %d steps: give each step its input, operation and output" % len(SEQ.findall(raw)), s["id"])
    for cm in CONFIRM.finditer(text):
        issue(iss, "needs_author_confirmation", "info", "open for the author: %s" % cm.group(1).strip(), None, item=cm.group(1).strip())
    # ---- limitations
    lim = [s for s in sents if part_role(s["part"]) == "limitations"]
    limcue = [s for s in sents if GENERIC_LIMIT.search(s["text"]) or re.search(r"\blimitation|局限|不足", s["text"], re.I)]
    cand_l = lim or limcue
    if not cand_l:
        issue(iss, "no_limitations", "warn", "no limitations are stated: name the settings in which the method or the comparison does not hold", None)
    for s in cand_l:
        if GENERIC_LIMIT.search(s["text"]) and not SCENARIO.search(s["text"]):
            issue(iss, "generic_limitation", "warn", "generic limitation (\"%s\"): tie it to a failure scenario (input type, size, range, setting not tested)" % s["text"][:70], s["id"])
    for l in ops.get("limitations", []):
        pass
    # ---- elements
    els = F.elements(allf, M["commands"], CONFIRM.sub("", text))
    for e in els:
        if e["status"] == "omitted_in_text":
            issue(iss, "element_omitted", "warn", "%s is in the materials (%s) but the text does not give it" % (e["label"], ", ".join(e["material"][:2])), None, element=e["id"])
        elif e["status"] == "text_without_source":
            issue(iss, "element_without_source", "warn", "the text mentions %s but no file / log / note documents it" % e["label"], None, element=e["id"])
    unknown = [e for e in els if e["status"] == "unknown_from_material" and not any(c["name"] == e["id"] for c in ops.get("confirms", []))]
    for e in unknown:
        if not re.search(re.escape(e["id"].split(".")[-1].replace("_", " ")) + r"|\[\[CONFIRM", text):
            pass
    marked = {m.group(1).strip().lower() for m in CONFIRM.finditer(text)}
    for e in unknown:
        keys = F.ELEM_KEYS.get(e["id"], []) + [e["id"], e["id"].split(".")[-1].replace("_", " ")]
        mentioned = any(k in mk for mk in marked for k in keys)
        if not mentioned:
            issue(iss, "gap_not_marked", "warn", "%s is nowhere in the materials and the text does not flag it: add [[CONFIRM: %s]] where it belongs" % (e["label"], e["label"].split(" (")[0]), None, element=e["id"])
    # ---- conflicts left
    for c in open_conflicts(A["conflicts"], res):
        issue(iss, "open_conflict", "warn", "sources disagree on %s [%s]: %s" % (c["canon"], c["scope"], "  vs  ".join("%s @ %s" % (F.show(v["value"]), v["sources"][0]) for v in c["values"])), None, canon=c["canon"])
    # ---- length
    n = len(re.findall(r"[㐀-鿿]", text)) if nlib.cjk_ratio(text) > 0.3 else len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))
    ln = M["task"].get("length")
    if ln and (n < 0.7 * ln or n > 1.4 * ln):
        issue(iss, "length", "warn", "the section has about %d %s; the task asks for about %d" % (n, "characters" if nlib.cjk_ratio(text) > 0.3 else "words", ln), None)
    sev = Counter(i["severity"] for i in iss)
    return {"issues": iss, "numbers": rows, "elements": els, "stats": {"sentences": len(sents), "numbers": len(rows), "sourced": sum(1 for r in rows if r["status"].startswith(("sourced", "derived"))), "severity": dict(sev), "words": n}, "sentences": sents}


def pick_source(cands, stage):
    pref = [f for f in cands if f["scope"] == stage] if stage else []
    pool = pref or cands
    return sorted(pool, key=lambda f: -F.TRUST.get(f["kind"], 1))[0]


def canon_pkg(n):
    n = n.lower().replace("_", "-")
    return {"pytorch": "torch", "scikit-learn": "sklearn", "opencv-python": "opencv"}.get(n, n)


def cmd_check(a):
    M, ops = load(a.work)
    res = check_draft(M, ops, rtext(a.draft))
    jsave(os.path.join(a.work, "check.json"), {k: v for k, v in res.items() if k != "sentences"})
    st = res["stats"]
    print("draft: %d sentences, %d numbers (%d tied to a source) | issues: %s" % (st["sentences"], st["numbers"], st["sourced"], st["severity"]))
    order = {"error": 0, "warn": 1, "info": 2}
    for i in sorted(res["issues"], key=lambda x: (order[x["severity"]], x["code"], x["sentence"] or "")):
        if i["severity"] == "info" and not getattr(a, "all", False):
            continue
        print("  %-5s %-26s %s%s" % (i["severity"], i["code"], (i["sentence"] + " ") if i["sentence"] else "", i["message"][:230]))
    cov = Counter(e["status"] for e in res["elements"])
    print("elements: %s" % dict(cov))
    if st["severity"].get("error"):
        print("Fix the errors (a number without a source is removed or asked for, never kept for the sake of the prose), then `check` again.")


# ------------------------------------------------------------------------------------------------ skeleton
def cmd_skeleton(a):
    M, ops = load(a.work)
    A = analysis(M, ops)
    T = {r["canon"] + "|" + r["scope"]: r for r in A["table"] if r["canon"]}

    def val(canon, *scopes):
        for sc in scopes + ("general",):
            r = T.get("%s|%s" % (canon, sc))
            if r:
                return r
        return None

    def txt(r):
        if r is None:
            return None
        if r["status"] == "CONFLICT":
            return "[[CONFIRM: %s - sources disagree: %s]]" % (r["name"], r["value"])
        v = r["value"]
        return F.show(v) if not isinstance(v, list) else ", ".join(F.show(x) for x in v)
    lines = ["## Experimental setup", ""]
    els = {e["id"]: e for e in F.elements(M["facts"], M["commands"], "")}

    def need(eid):
        return "[[CONFIRM: %s]]" % els[eid]["label"].split(" (")[0]
    sent = []
    ep, lr, bs, opt = val("epochs", "train"), val("learning_rate", "train"), val("batch_size", "train"), val("optimizer", "train")
    sent.append("Models were trained for %s epochs with %s (learning rate %s, batch size %s)." % (txt(ep) or need("setup.epochs"), txt(opt) or need("setup.optimizer"), txt(lr) or need("setup.learning_rate"), txt(bs) or need("setup.batch_size")))
    ebs = val("batch_size", "test")
    if ebs:
        sent.append("Evaluation used a batch size of %s." % txt(ebs))
    sd = val("seed", "train")
    sent.append("The random seed was %s." % (txt(sd) or need("setup.seed")))
    gpu = [f for f in M["facts"] if f["name"] == "hardware.gpu"]
    ng = [f for f in M["facts"] if f["name"] == "hardware.num_gpus"]
    sent.append("Experiments ran on %s%s." % ("%s " % F.show(ng[0]["value"]) if ng else "", gpu[0]["value"] if gpu else need("setup.hardware")))
    env = [f for f in M["facts"] if f["name"] in ("env.torch", "env.pytorch", "env.cuda", "env.python")]
    sent.append("Software: %s." % ("; ".join("%s %s" % (f["name"][4:], f["value"]) for f in env[:4]) if env else need("setup.software")))
    lines += [" ".join(sent), "", "## Dataset", ""]
    ds = {f["name"]: f for f in M["facts"] if f["name"].startswith(("dataset.", "data."))}
    name = next((f for f in M["facts"] if re.search(r"(?:^|\.)(?:dataset|data)\.name$", f["name"])), None)
    ver = next((f for f in M["facts"] if re.search(r"version$", f["name"], re.I) and f["scope"] != "env"), None)
    sizes = [(f["name"].split(".")[-1], f["value"]) for f in M["facts"] if f["name"].startswith("dataset.size.")]
    sp = next((f for f in M["facts"] if f["name"] == "dataset.split_ratio" or f["name"] == "data.split"), None)
    lines.append("%s%s. %s %s" % (name["value"] if name else need("data.name"), (" " + str(ver["value"])) if ver else " " + need("data.version"), ("Sizes: " + ", ".join("%s %s" % (k, F.show(v)) for k, v in sizes) + ".") if sizes else need("data.size"), ("Split: %s." % (sp["value"],)) if sp else need("data.split")))
    lines += ["", "## Baselines and evaluation", "", need("baseline.statistics") + " Metrics: " + (", ".join("%s %s" % (k, F.show(v[-1]["value"])) for k, v in list(A["results"].items())[:6]) or need("baseline.metrics")) + ".", "",
              "## Reproducibility", "", ("Commands: " + "; ".join("`%s`" % c["command"][:80] for c in M["commands"][:3]) + ".") if M["commands"] else need("repro.commands"), need("repro.code_version"), need("repro.data_access"), "",
              "## Limitations", "", "[[CONFIRM: concrete failure scenarios - when does the method or the comparison not hold?]]", ""]
    out = "\n".join(lines)
    if a.out:
        wtext(a.out, out)
        print("skeleton written to %s" % a.out)
    else:
        print(out)


# ------------------------------------------------------------------------------------------------ build
def contract_fields(case):
    oc = find_key(case, ["output_contract", "outputContract", "output_schema", "outputs"]) if case else None
    items = oc if isinstance(oc, list) else (oc.get("fields") or oc.get("required") or oc.get("properties") or oc.get("required_fields") or [k for k in oc if k not in ("type", "description", "title", "version", "schema", "$schema", "additionalProperties")]) if isinstance(oc, dict) else []
    if isinstance(items, dict):
        items = list(items)
    out = []
    for it in items:
        n = it if isinstance(it, str) else (it.get("name") or it.get("field") or it.get("key")) if isinstance(it, dict) else None
        if isinstance(n, str) and n not in out:
            out.append(n)
    return out


def derive_field(name, ctx):
    n = name.lower()
    words = set(re.split(r"[^a-z]+", n))

    def has(ascii_words=(), cjk=()):
        return bool(words & set(ascii_words)) or any(x in n for x in cjk)
    if n in ("status", "run_status", "state", "outcome") or has(("status",), ("状态",)):
        return True, ctx["status"]
    if has(("product", "manuscript", "deliverable", "target", "draft", "section", "text"), ("产物", "稿", "章节")):
        return True, {"files": [f for f in ctx["files"] if f.startswith("section")], "text": ctx["text"], "word_count": ctx["words"]}
    if has(("parameters", "params", "hyperparameters"), ("参数",)):
        return True, ctx["table"]
    if has(("dataset", "datasets", "data"), ("数据集",)):
        return True, ctx["datasets"]
    if has(("source", "sources", "evidence", "provenance", "mapping"), ("来源", "映射")):
        return True, {"numbers": ctx["numbers"], "parameters": ctx["table"], "elements": ctx["elements"], "environment": ctx["env"]}
    if has(("boundary", "boundaries", "limits", "limitations", "uncertainty", "uncertainties", "caveats", "questions", "review"), ("边界", "版本", "不确定", "确认", "缺口")):
        return True, {"questions_for_author": ctx["questions"], "open_issues": ctx["open"], "conflicts": ctx["conflicts"], "uncertainty": ctx["uncertainty"]}
    if has(("files", "artifacts"), ()):
        return True, ctx["files"]
    return False, None


def cmd_build(a):
    M, ops = load(a.work)
    text = rtext(a.draft)
    A = analysis(M, ops)
    chk = check_draft(M, ops, text, A)
    os.makedirs(a.out, exist_ok=True)
    files = []
    final = tidy(text)
    wtext(os.path.join(a.out, "section.md"), final)
    blocks = []
    for b in re.split(r"\n\s*\n", final.strip()):
        mh = re.match(r"^(#{1,6})\s+(.+?)\s*$", b.strip())
        if mh and "\n" not in b.strip():
            blocks.append(("h1" if len(mh.group(1)) <= 2 else "h2", mh.group(2)))
        else:
            blocks.append(("p", re.sub(r"\s*\n\s*", " ", b.strip())))
    docxw.make(blocks, os.path.join(a.out, "section.docx"))
    tex = []
    for b in blocks:
        tex.append(("\\%ssection{%s}" % ("sub" * (0 if b[0] == "h1" else 1), b[1])) if b[0] != "p" else b[1].replace("%", "\\%").replace("&", "\\&").replace("_", "\\_"))
    wtext(os.path.join(a.out, "section.tex"), "\n\n".join(tex) + "\n")
    files += ["section.md", "section.docx", "section.tex"]
    # tables
    with open(os.path.join(a.out, "parameters.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["parameter", "value", "scope", "status", "sources"])
        for r in A["table"]:
            w.writerow([r["name"], F.show(r["value"]) if not isinstance(r["value"], list) else json.dumps(r["value"]), r["scope"], r["status"], "; ".join(r["sources"])])
    md = ["| parameter | value | scope | status | source |", "|---|---|---|---|---|"] + ["| %s | %s | %s | %s | %s |" % (r["name"], str(r["value"]).replace("|", "/"), r["scope"], r["status"], "; ".join(r["sources"][:2])) for r in A["table"]]
    wtext(os.path.join(a.out, "parameters.md"), "\n".join(md) + "\n")
    with open(os.path.join(a.out, "datasets_environment.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "value", "source"])
        for r in A["datasets"] + A["env"]:
            w.writerow([r["item"], r["value"], r["source"]])
    wtext(os.path.join(a.out, "commands.md"), "# Commands found in the materials\n\n" + ("\n".join("- `%s` (%s:%s)" % (c["command"], c["file"], c["line"]) for c in M["commands"]) or "- none found") + "\n")
    with open(os.path.join(a.out, "source_map.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sentence", "number", "kind", "status", "sources"])
        for r in chk["numbers"]:
            w.writerow([r["sentence"], r["token"], r["kind"], r["status"], "; ".join(r["sources"])])
    jsave(os.path.join(a.out, "source_map.json"), {"numbers": chk["numbers"], "elements": chk["elements"], "parameters": A["table"], "conflicts": A["conflicts"], "resolutions": ops["resolutions"], "confirmations": ops.get("confirms", [])})
    files += ["parameters.csv", "parameters.md", "datasets_environment.csv", "commands.md", "source_map.csv", "source_map.json"]
    # questions
    questions = []
    for c in open_conflicts(A["conflicts"], ops["resolutions"]):
        questions.append({"kind": "conflict", "item": c["canon"], "question": "Which value is right for %s [%s]? %s. (%s)" % (c["canon"], c["scope"], "  vs  ".join("%s in %s" % (F.show(v["value"]), ", ".join(v["sources"][:2])) for v in c["values"]), c["hint"])})
    for e in chk["elements"]:
        if e["status"] == "unknown_from_material" and not any(cf["name"] == e["id"] for cf in ops.get("confirms", [])):
            questions.append({"kind": "missing", "item": e["id"], "question": F.QUESTION[e["id"]]})
    for i in chk["issues"]:
        if i["code"] in ("unsourced_number", "unsourced_name", "unsourced_version", "element_without_source", "unverified_comparability"):
            questions.append({"kind": i["code"], "item": i.get("token") or i.get("element") or "", "question": i["message"]})
        if i["code"] == "needs_author_confirmation" and not any(q["kind"] == "missing" and i["item"].lower() in q["question"].lower() for q in questions):
            questions.append({"kind": "marked_in_text", "item": i["item"], "question": i["item"]})
    wtext(os.path.join(a.out, "questions.md"), "# Questions for the author (the materials do not answer these)\n\n" + ("\n".join("%d. [%s] %s" % (n, q["kind"], q["question"]) for n, q in enumerate(questions, 1)) or "No open question.") + "\n")
    files.append("questions.md")
    errs = [i for i in chk["issues"] if i["severity"] == "error"]
    warns = [i for i in chk["issues"] if i["severity"] == "warn"]
    unc = []
    unc.append({"kind": "extraction_is_textual", "detail": "values are read from the files' text (config keys, argparse defaults, call keywords, log lines, prose): a value computed at run time, or set by an environment variable or a code path the files do not show, is not visible"})
    if open_conflicts(A["conflicts"], ops["resolutions"]):
        unc.append({"kind": "open_conflicts", "detail": "%d parameter(s) have conflicting values across sources; the text may not state them as settled" % len(open_conflicts(A["conflicts"], ops["resolutions"]))})
    if not M["facts"]:
        unc.append({"kind": "no_facts", "detail": "no parameters, logs or notes were found: the section can only restate what the task text gives"})
    status = "success" if not errs and not open_conflicts(A["conflicts"], ops["resolutions"]) and M["facts"] else "partial"
    ctx = {"status": status, "text": final, "words": chk["stats"]["words"], "files": files, "table": A["table"], "datasets": A["datasets"], "env": A["env"], "numbers": chk["numbers"], "elements": chk["elements"], "questions": questions,
           "open": [{"code": i["code"], "severity": i["severity"], "sentence": i["sentence"], "message": i["message"]} for i in chk["issues"] if i["severity"] != "info"], "conflicts": A["conflicts"], "uncertainty": unc}
    review = ["- [ ] %s %s: %s" % (i["code"], i["sentence"] or "", i["message"][:260]) for i in errs + warns]
    wtext(os.path.join(a.out, "review_checklist.md"), "# Review checklist\n\nMachine-assisted draft: nothing here is approved by the author.\n\n" + ("\n".join(review) or "- no open item") + "\n\n## Always confirm by hand\n- every value marked CONFLICT or `[[CONFIRM: ...]]`\n- that the comparison conditions stated for the baselines were really the same\n- values the files cannot show (set at run time, by environment variable, by an unshown code path)\n")
    files.append("review_checklist.md")
    case = M.get("case")
    fields = contract_fields(case)
    answers = jload(a.answers, {}) if a.answers else {}
    result = {"case_id": M.get("case_id")}
    unfilled = []
    for fld in fields:
        if fld == "case_id":
            continue
        if fld in answers:
            result[fld] = answers[fld]
            continue
        ok, v = derive_field(fld, ctx)
        result[fld] = v if ok else None
        if not ok:
            unfilled.append(fld)
    for nm, key in (("status", "status"), ("manuscript", "manuscript"), ("sources", "sources"), ("boundaries", "boundaries"), ("files", "files")):
        if nm not in result:
            result[nm] = derive_field({"status": "status", "manuscript": "section", "sources": "source", "boundaries": "boundaries", "files": "files"}[key], ctx)[1]
    for k, v in answers.items():
        result.setdefault(k, v)
    result["_meta"] = {"generator": "methods-ledger/mw.py %s" % VERSION, "contract_fields": fields, "unfilled_fields": unfilled, "contract_found": bool(fields), "files": files}
    jsave(os.path.join(a.out, "result.json"), result)
    st = chk["stats"]
    print(json.dumps({"written": a.out, "status": status, "numbers": st["numbers"], "numbers_tied_to_a_source": st["sourced"], "issues": st["severity"], "open_conflicts": len(open_conflicts(A["conflicts"], ops["resolutions"])), "questions": len(questions), "unfilled_contract_fields": unfilled, "contract_found": bool(fields)}, ensure_ascii=False))
    for e in errs[:10]:
        print("OPEN ERROR %s %s: %s" % (e["code"], e["sentence"] or "", e["message"][:170]))
    if unfilled:
        print("UNFILLED contract fields (put them in answers.json and re-run build --answers): " + ", ".join(unfilled))
    if getattr(a, "case", None) is not None and not fields and a.case:
        print("WARNING: no output_contract found in the case file: read /input/case.json and task.md and add the fields by hand via --answers")
    sys.exit(0 if (not unfilled or not fields) else 2)


def tidy(text):
    t = re.sub(r"[ \t]+([,.;:。，；：)])", r"\1", text)
    t = re.sub(r"(?m)[ \t]+$", "", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip() + "\n"


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(prog="mw.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.set_defaults(fn=cmd_run)
    p.add_argument("--case")
    p.add_argument("--src", action="append")
    p.add_argument("--file", action="append")
    p.add_argument("--out", required=True)
    p.add_argument("--work")
    p = sub.add_parser("show")
    p.set_defaults(fn=cmd_show)
    p.add_argument("work")
    p.add_argument("--facts")
    p.add_argument("--conflicts", action="store_true")
    p.add_argument("--results", action="store_true")
    p.add_argument("--limit", type=int, default=80)
    p = sub.add_parser("skeleton")
    p.set_defaults(fn=cmd_skeleton)
    p.add_argument("work")
    p.add_argument("--out")
    p = sub.add_parser("ingest")
    p.set_defaults(fn=cmd_ingest)
    p.add_argument("work")
    p.add_argument("file", nargs="?")
    p = sub.add_parser("check")
    p.set_defaults(fn=cmd_check)
    p.add_argument("work")
    p.add_argument("draft")
    p.add_argument("--all", action="store_true")
    p = sub.add_parser("build")
    p.set_defaults(fn=cmd_build)
    p.add_argument("work")
    p.add_argument("draft")
    p.add_argument("--out", required=True)
    p.add_argument("--case", default="")
    p.add_argument("--answers")
    a = ap.parse_args(argv)
    import zipfile
    import xml.etree.ElementTree as ET
    try:
        a.fn(a)
    except zipfile.BadZipFile:
        raise SystemExit("ERROR: an input is not a valid .docx (corrupt or truncated)")
    except ET.ParseError as e:
        raise SystemExit("ERROR: a Word document's XML cannot be read (%s)" % e)
    except FileNotFoundError as e:
        raise SystemExit("ERROR: file not found: %s" % e.filename)


if __name__ == "__main__":
    main()
