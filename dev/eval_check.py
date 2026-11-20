#!/usr/bin/env python3
"""eval_check.py - does `check` accept sentences that match the sources and reject sentences that do not?

Statements are generated from the facts of the invented project in dev/fixtures/project (one sentence per canonical parameter, in 3 phrasings and in Chinese):
  faithful   : the value the sources give (settled values only)
  perturbed  : the same sentence with the value changed (x2, +1 or a different digit)
  invented   : a sentence about a parameter that has no source at all in the project (momentum, warm-up ratio, gradient clipping)
Counted per statement: error issues raised for that sentence. This measures the number-to-source matching, not the quality of a real draft.
Usage: python3 dev/eval_check.py
"""
import contextlib
import io
import json
import os
import random
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, next(p for p in (os.path.join(HERE, "..", "submission", "scripts"), os.path.join(HERE, "..", "scripts")) if os.path.isdir(p)))
import mw  # noqa: E402

PROJ = os.path.join(HERE, "fixtures", "project")
tmp = tempfile.mkdtemp()
with contextlib.redirect_stdout(io.StringIO()):
    try:
        mw.main(["run", "--src", PROJ, "--out", tmp])
    except SystemExit:
        pass
M, ops = mw.load(os.path.join(tmp, ".work"))
A = mw.analysis(M, ops)
rows = {(r["canon"], r["scope"]): r for r in A["table"] if r["canon"] and r["status"] in ("single-source", "consistent") or (r["canon"] and r["status"].startswith("settled"))}
T_EN = {"learning_rate": ["The learning rate was %s.", "We set the learning rate to %s.", "Training used a learning rate of %s."],
        "batch_size": ["The batch size was %s.", "We used a batch size of %s.", "Each step processes batches of %s samples."],
        "epochs": ["Training lasted %s epochs.", "We trained for %s epochs.", "The model was trained for %s epochs."],
        "seed": ["The random seed was %s.", "We fixed the seed to %s.", "All runs used seed %s."],
        "weight_decay": ["The weight decay was %s.", "We used a weight decay of %s.", "Weight decay is set to %s."],
        "image_size": ["Images were resized to %s pixels.", "The input resolution is %s.", "We crop images to %s."],
        "warmup": ["We used %s warm-up steps.", "The warm-up lasts %s steps.", "A warm-up of %s steps precedes cosine decay."]}
T_ZH = {"learning_rate": ["学习率为%s。"], "batch_size": ["批大小为%s。"], "epochs": ["训练%s个epoch。"], "seed": ["随机种子为%s。"]}
INVENTED = ["The momentum was 0.9.", "We clipped gradients at norm 1.0.", "The warm-up ratio was 0.06.", "We used label smoothing of 0.1.", "The temperature was 0.07.", "Training used 3 random restarts."]
rng = random.Random(5)


def errors_for(sentences):
    p = os.path.join(tmp, "d.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("## Setup\n\n" + "\n\n".join(sentences) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        res = mw.check_draft(M, ops, open(p, encoding="utf-8").read())
    by = {}
    for i in res["issues"]:
        if i["severity"] == "error" and i["sentence"]:
            by.setdefault(i["sentence"], []).append(i["code"])
    return [by.get("D%03d" % (k + 1), []) for k in range(len(sentences))]


def fmtv(v):
    return ("%g" % v) if isinstance(v, float) else str(v)


def perturb(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        c = rng.choice([v * 2, v + 1, v * 10, v / 2])
        return c if c != v else v + 3
    return None


faith, pert, fa_ok, pe_det = [], [], 0, 0
cases = []
for (canon, scope), r in rows.items():
    v = r["value"]
    if canon in T_EN and isinstance(v, (int, float)) and not isinstance(v, bool):
        for t in T_EN[canon]:
            cases.append((t, v))
        if canon in T_ZH:
            for t in T_ZH[canon]:
                cases.append((t, v))
faithful_s = [t % fmtv(v) for t, v in cases]
perturbed_s = [t % fmtv(perturb(v)) for t, v in cases]
fe = errors_for(faithful_s)
pe = errors_for(perturbed_s)
ie = errors_for(INVENTED)
print("faithful statements: %d, accepted without an error: %d (%d%%)" % (len(fe), sum(1 for e in fe if not e), 100 * sum(1 for e in fe if not e) // max(1, len(fe))))
print("perturbed statements: %d, flagged: %d (%d%%)" % (len(pe), sum(1 for e in pe if e), 100 * sum(1 for e in pe if e) // max(1, len(pe))))
print("invented parameters (no source): %d, flagged: %d (%d%%)" % (len(ie), sum(1 for e in ie if e), 100 * sum(1 for e in ie if e) // max(1, len(ie))))
for s, e in zip(faithful_s, fe):
    if e:
        print("  false alarm:", s, e)
for s, e in zip(perturbed_s, pe):
    if not e:
        print("  missed:", s)
for s, e in zip(INVENTED, ie):
    if not e:
        print("  invented but accepted:", s)
shutil.rmtree(tmp, ignore_errors=True)
