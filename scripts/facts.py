#!/usr/bin/env python3
"""facts.py - from facts to what a methods section needs: conflicts between sources, the parameter / dataset / environment tables, and the checklist of elements.

A conflict is never settled silently: two sources that give different values for the same parameter in the same scope are listed with their files and lines until the author's decision
(`=== resolve`) is recorded. The tables carry the resolved value, or both values marked as conflict.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import extract

RUNTIME = ("log", "cli", "result")
TRUST = {"cli": 5, "log": 5, "config": 4, "argparse": 3, "call": 3, "assign": 3, "doc": 2, "env": 4, "result": 5, "data": 3}
STAGES = ("train", "val", "test")


def vkey(v):
    """comparable form of a value"""
    if isinstance(v, bool):
        return "b:%s" % v
    if isinstance(v, (int, float)):
        return "n:%.10g" % float(v)
    if isinstance(v, list):
        return "l:" + ",".join(vkey(x) for x in v)
    n = extract.to_num(v) if isinstance(v, str) else None
    if n is not None and not str(v).endswith("%"):
        return "n:%.10g" % n
    return "s:" + re.sub(r"[\s_\-]+", "", str(v).lower())


def show(v):
    if isinstance(v, float):
        return ("%.10g" % v)
    return str(v)


def compatible_scopes(a, b):
    if a == b:
        return True
    gen = ("general", "optim", "model", "data")
    return (a in gen and b in gen) or (a in gen and b not in STAGES) or (b in gen and a not in STAGES)


TRAINISH = {"learning_rate", "epochs", "optimizer", "scheduler", "weight_decay", "momentum", "warmup", "steps", "grad_clip", "patience", "gamma", "dropout", "beta", "eps", "loss", "batch_size"}


def bucket(f):
    """where a fact lives in the tables: stage-sensitive training parameters by stage (a fact without a stage is a training value); every other parameter in one 'general' row"""
    if f["canon"] in TRAINISH:
        return f["scope"] if f["scope"] in STAGES else "train"
    return "general"


def ckey(canon, v):
    """comparable form; scheduler / optimizer / model names that contain each other (cosine / CosineAnnealingLR) are the same"""
    if canon in ("scheduler", "optimizer", "model", "loss") and isinstance(v, str):
        s = re.sub(r"[^a-z0-9]", "", v.lower())
        s = re.sub(r"annealing|scheduler|lr$|decay", "", s)
        return "s:" + s
    return vkey(v)


TIER_A = ("log", "cli")


def tiers(group, canon):
    """split the facts of one bucket by how they came about"""
    A = [f for f in group if f["kind"] in TIER_A]
    B = [f for f in group if f["kind"] == "config" or f["kind"] == "confirm"]
    C = [f for f in group if f["kind"] in ("call", "assign", "doc")]
    D = [f for f in group if f["kind"] == "argparse"]
    return A, B, C, D


def settle(group, canon):
    """-> ("conflict" | "settled" | "single", chosen_value, note, losers) for one bucket; runtime evidence (log / command line) settles what the files only default; argparse defaults never conflict with a config"""
    A, B, C, D = tiers(group, canon)
    vals = lambda fs: {ckey(canon, f["value"]): f for f in fs}
    va, vb, vc, vd = vals(A), vals(B), vals(C), vals(D)
    if len(va) > 1:
        return "conflict", None, "the run logs / command lines disagree", []
    if len(va) == 1:
        keep = next(iter(va.values()))
        others = [f for f in B + C + D if ckey(canon, f["value"]) != ckey(canon, keep["value"])]
        return ("settled" if others else "single"), keep["value"], "run log / command line (%s)" % keep["source"] if others else "", others
    if len(vb) > 1:
        return "conflict", None, "configuration files disagree", []
    if len(vb) == 1:
        keep = next(iter(vb.values()))
        others = [f for f in C if ckey(canon, f["value"]) != ckey(canon, keep["value"])]
        if others:
            return "conflict", None, "a configuration file and a value fixed in code / notes disagree", []
        return "single", keep["value"], "", [f for f in D if ckey(canon, f["value"]) != ckey(canon, keep["value"])]
    allv = {**vc, **vd}
    if len(allv) > 1:
        return "conflict", None, "values in code / notes disagree and no config or log settles them", []
    return "single", (next(iter(allv.values()))["value"] if allv else None), "", []


def conflicts(facts):
    """-> list of {canon, scope, values: [{value, facts, sources, kinds}], runtime, hint}.
    One bucket per (canon, stage) plus one for facts without a stage (they join every stage bucket: they may apply to all of them). See settle() for what counts as a conflict."""
    by = defaultdict(list)
    for f in facts:
        if f["canon"] and f["value"] is not None and not isinstance(f["value"], dict):
            if f["name"].endswith(("_first", "_last", "epochs_logged")) or (re.search(r"\[\d+\]$", f["name"]) and f["canon"] != "seed"):
                continue
            if isinstance(f["value"], list) and f["canon"] not in ("seed", "test_size", "image_size", "beta", "repeats"):
                continue
            if f["canon"] == "loss" and isinstance(f["value"], (int, float)):
                continue
            by[f["canon"]].append(f)
    out, seen = [], set()
    for canon, fs in by.items():
        buckets = defaultdict(list)
        for x in fs:
            buckets[bucket(x)].append(x)
        for scope, group in buckets.items():
            st, chosen, note, losers = settle(group, canon)
            if st != "conflict":
                continue
            vals = defaultdict(list)
            for x in group:
                vals[ckey(canon, x["value"])].append(x)
            sig = (canon, scope, tuple(sorted(vals)))
            if sig in seen:
                continue
            seen.add(sig)
            rows = []
            for k, fl in vals.items():
                rows.append({"value": fl[0]["value"], "facts": [x["id"] for x in fl], "sources": sorted({x["source"] for x in fl}), "kinds": sorted({x["kind"] for x in fl}), "best_trust": max(TRUST.get(x["kind"], 1) for x in fl)})
            rows.sort(key=lambda r: -r["best_trust"])
            out.append({"canon": canon, "scope": scope, "values": rows, "runtime": any(set(r["kinds"]) & set(TIER_A) for r in rows), "hint": note})
    return out


def resolution_for(res, canon, scope):
    for r in res:
        if r["canon"] == canon and r["scope"] in (scope, "any", ""):
            return r
    return None


# ------------------------------------------------------------------------------------------------ tables
def param_table(facts, confl, resolutions):
    """rows {name, canon, value, scope, sources, status, facts}; status: single-source | consistent | resolved | CONFLICT"""
    cmap = {(c["canon"], c["scope"]): c for c in confl}
    rows, used = [], set()
    groups = defaultdict(list)
    for f in facts:
        if f["kind"] in ("result", "env") or f["scope"] == "env" or f["name"].endswith(("_first", "_last", "epochs_logged")):
            continue
        if f["kind"] == "data" or f["name"].startswith(("dataset.", "table_rows.")):
            continue
        if isinstance(f["value"], list) and not f["canon"]:
            continue
        if re.search(r"\[\d+\]$", f["name"]):
            continue
        if isinstance(f["value"], str) and re.search(r"^[\w./\\:-]+\.(?:ya?ml|json|py|pt|pth|ckpt|txt|csv|npz?)$", f["value"]) or re.search(r"(?:^|\.)(?:root|path|dir|ckpt|checkpoint|config|output_dir|data_root|log_dir)$", f["name"], re.I):
            continue
        if f["kind"] == "doc" and not f["canon"]:
            continue
        if f["kind"] in ("call", "assign") and not f["canon"]:
            continue
        key = (f["canon"] or f["name"], bucket(f) if f["canon"] else (f["scope"] if f["scope"] in STAGES else "general"))
        groups[key].append(f)
    for (name, scope), fl in sorted(groups.items(), key=lambda kv: (str(kv[0][1]), str(kv[0][0]))):
        canon = fl[0]["canon"]
        res = resolution_for(resolutions, name, scope)
        vals = defaultdict(list)
        for f in fl:
            vals[ckey(canon, f["value"])].append(f)
        f0 = fl[0]
        if len(vals) == 1:
            rows.append({"name": f0["name"] if not canon else name, "value": f0["value"], "scope": scope, "sources": sorted({f["source"] for f in fl}), "status": "consistent" if len({f["source"].split(":")[0] for f in fl}) > 1 else "single-source", "kinds": sorted({f["kind"] for f in fl}), "facts": [f["id"] for f in fl], "canon": canon})
            continue
        if res:
            fl_sel = [f for f in fl if ckey(canon, f["value"]) == ckey(canon, res["value"])]
            rows.append({"name": name, "value": res["value"], "scope": scope, "sources": sorted({f["source"] for f in fl_sel}) or ["author decision"], "status": "resolved (%s)" % (res["reason"][:60] or res["source"]), "kinds": sorted({f["kind"] for f in fl_sel}) or ["author"], "facts": [f["id"] for f in fl_sel], "canon": name,
                         "others": [show(f["value"]) + " @ " + f["source"] for f in fl if ckey(canon, f["value"]) != ckey(canon, res["value"])]})
            continue
        st, chosen, note, losers = settle(fl, canon) if canon else ("conflict", None, "", [])
        if st in ("settled", "single") and chosen is not None:
            fl_sel = [f for f in fl if ckey(canon, f["value"]) == ckey(canon, chosen)]
            lose = [f for f in fl if ckey(canon, f["value"]) != ckey(canon, chosen)]
            rows.append({"name": name, "value": chosen, "scope": scope, "sources": sorted({f["source"] for f in fl_sel}), "status": ("settled by " + note) if st == "settled" else "single-source" if len({f["source"].split(":")[0] for f in fl_sel}) == 1 else "consistent",
                         "kinds": sorted({f["kind"] for f in fl_sel}), "facts": [f["id"] for f in fl_sel], "canon": canon, "others": ["%s @ %s (overridden default)" % (show(f["value"]), f["source"]) for f in lose]})
            continue
        rows.append({"name": name, "value": " | ".join("%s (%s)" % (show(v[0]["value"]), v[0]["source"]) for v in sorted(vals.values(), key=lambda v: -max(TRUST.get(f["kind"], 1) for f in v))), "scope": scope, "sources": sorted({f["source"] for f in fl}), "status": "CONFLICT", "kinds": sorted({f["kind"] for f in fl}), "facts": [f["id"] for f in fl], "canon": canon})
    return rows


def dataset_rows(facts):
    rows = []
    for f in facts:
        if f["name"].startswith("dataset.") or f["name"].startswith("data.") and f["canon"] == "" and f["kind"] in ("config",) and re.search(r"name|version|root|path|split|size", f["name"]):
            rows.append({"item": f["name"], "value": f["value"], "source": f["source"], "kind": f["kind"], "id": f["id"]})
    for f in facts:
        if f["name"].startswith("table_rows."):
            rows.append({"item": f["name"], "value": f["value"], "source": f["source"], "kind": f["kind"], "id": f["id"]})
    return rows


def env_rows(facts):
    rows, seen = [], set()
    for f in facts:
        if f["scope"] == "env" or f["kind"] == "env":
            k = (f["name"], vkey(f["value"]))
            if k in seen:
                continue
            seen.add(k)
            rows.append({"item": f["name"], "value": f["value"], "source": f["source"], "kind": f["kind"], "id": f["id"]})
    return rows


def results_rows(facts):
    """last value per metric name (and every source) for the brief"""
    by = defaultdict(list)
    for f in facts:
        if f["kind"] == "result":
            by[f["name"]].append(f)
    return {k: v for k, v in by.items()}


# ------------------------------------------------------------------------------------------------ elements checklist
# (id, category, label, material detector over facts, phrase regex for the text)
def has_name(facts, *names):
    return [f for f in facts if any(f["name"].lower().endswith(n) or f["canon"] == n for n in names)]


ELEMENTS = [
    ("setup.seed", "setup", "random seed(s)", lambda F: [f for f in F if f["canon"] == "seed"], r"random seeds?|\bseeds?\b|随机种子|种子"),
    ("setup.repeats", "setup", "number of repeated runs / folds (or a statement that there was one run)", lambda F: [f for f in F if f["canon"] == "repeats"] + [f for f in F if f["canon"] == "seed" and isinstance(f["value"], list) and len(f["value"]) > 1], r"repeated|independent runs|runs|repeats|folds?|trials|重复|次实验|折交叉|一次运行|single run"),
    ("setup.epochs", "setup", "training length (epochs / steps)", lambda F: [f for f in F if f["canon"] in ("epochs", "steps")], r"epochs?|steps|iterations|训练轮|迭代"),
    ("setup.batch_size", "setup", "batch size", lambda F: [f for f in F if f["canon"] == "batch_size"], r"batch[ -]size|mini-?batch|批大小|批量"),
    ("setup.optimizer", "setup", "optimizer", lambda F: [f for f in F if f["canon"] == "optimizer"], r"optimi[sz]er|\b(?:AdamW|Adam|SGD|RMSprop|Adagrad)\b|优化器"),
    ("setup.learning_rate", "setup", "learning rate (and schedule)", lambda F: [f for f in F if f["canon"] == "learning_rate"], r"learning[ -]rate|\blr\b|学习率"),
    ("setup.hardware", "setup", "hardware (GPU / CPU model and count)", lambda F: [f for f in F if f["name"].startswith("hardware.")], r"\bGPUs?\b|\bCPU\b|A100|V100|H100|RTX|显卡|处理器"),
    ("setup.software", "setup", "software versions (framework, CUDA, Python)", lambda F: [f for f in F if f["name"].startswith("env.") and f["name"] != "env.docker_image"], r"PyTorch|TensorFlow|CUDA|Python \d|JAX|版本|version"),
    ("data.name", "dataset", "dataset name", lambda F: [f for f in F if re.search(r"(?:^|\.)(?:dataset|data)\.(?:name|dataset)$|^dataset$|^dataset\.name$", f["name"])], None),
    ("data.version", "dataset", "dataset version", lambda F: [f for f in F if re.search(r"version$", f["name"], re.I) and f["scope"] != "env"], r"dataset version|data version|release|\bv\d+(?:\.\d+)+\b|数据集版本|数据版本"),
    ("data.size", "dataset", "dataset size (total and per split)", lambda F: [f for f in F if f["name"].startswith("dataset.size.")], r"samples|images|tiles|patches|examples|sentences|instances|records|样本|图像|条"),
    ("data.split", "dataset", "split proportions or sizes and how the split was drawn", lambda F: [f for f in F if f["name"] == "dataset.split_ratio" or f["canon"] == "test_size" or re.search(r"split", f["name"])], r"split|train/val|train-val|training, validation|划分|切分"),
    ("data.preprocessing", "dataset", "preprocessing / augmentation", lambda F: [f for f in F if f["canon"] == "image_size" or re.search(r"augment|normali[sz]|resize|crop|flip|mean|std|tokeni", f["name"], re.I)], r"preprocess|normali[sz]|resize|augment|crop|flip|tokeni|预处理|归一化|增强"),
    ("baseline.metrics", "baseline", "evaluation metrics and their values", lambda F: [f for f in F if f["kind"] == "result"], r"F1|accuracy|AUC|RMSE|MAE|IoU|BLEU|precision|recall|精度|准确率|指标"),
    ("baseline.statistics", "baseline", "statistics (mean ± std, number of runs, significance test)", lambda F: [f for f in F if re.search(r"std|stderr|p_value|ci_|confidence", f["name"], re.I)], r"±|standard deviation|std\b|confidence interval|p\s*[<=]|t-test|Wilcoxon|标准差|显著性"),
    ("repro.code_version", "repro", "code version (commit / tag)", lambda F: [f for f in F if re.search(r"commit|git|revision|tag$", f["name"], re.I) or (f["kind"] == "doc" and re.search(r"\bcommit\s+[0-9a-f]{7,40}\b", f["context"] or ""))], r"commit|tag\b|release|代码版本|git"),
    ("repro.commands", "repro", "commands to reproduce", None, r"python |bash |command|命令|脚本|run\.sh|torchrun"),
    ("repro.data_access", "repro", "how to obtain the data", lambda F: [f for f in F if re.search(r"https?://|doi\.org|zenodo|figshare|data_?(?:root|path|url|dir)|root$|download", (f["name"] + " " + str(f["value"])), re.I)], r"download|available at|https?://|doi\.org|zenodo|获取|下载|公开"),
]


def elements(facts, commands, text):
    """-> rows {id, category, label, material: [fact ids], text: bool, status}; status: covered | omitted_in_text | unknown_from_material | text_without_source"""
    rows = []
    for eid, cat, label, det, phrase in ELEMENTS:
        mat = det(facts) if det else []
        if eid == "repro.commands":
            mat = list(commands)
        in_text = bool(phrase and text and re.search(phrase, text, re.I))
        if eid == "data.name":
            in_text = bool(text) and bool(mat) and any(str(f["value"]).lower() in text.lower() for f in mat)
        if mat and in_text:
            st = "covered"
        elif mat and not in_text:
            st = "omitted_in_text" if text else "available"
        elif not mat and in_text:
            st = "text_without_source"
        else:
            st = "unknown_from_material"
        rows.append({"id": eid, "category": cat, "label": label, "material": [f["id"] if isinstance(f, dict) and "id" in f else "%s:%s" % (f.get("file"), f.get("line")) for f in mat][:6], "text": in_text, "status": st})
    return rows


ELEM_KEYS = {"setup.seed": ["seed", "种子"], "setup.repeats": ["repeat", "runs", "folds", "重复", "运行次数"], "setup.epochs": ["epoch", "steps", "轮"], "setup.batch_size": ["batch", "批"], "setup.optimizer": ["optimizer", "优化器"],
             "setup.learning_rate": ["learning rate", "schedule", "学习率"], "setup.hardware": ["hardware", "gpu", "cpu", "硬件", "显卡"], "setup.software": ["software", "version", "framework", "cuda", "python", "软件", "版本"],
             "data.name": ["dataset name", "数据集名"], "data.version": ["dataset version", "version", "release", "版本"], "data.size": ["size", "samples", "规模", "样本"], "data.split": ["split", "划分"], "data.preprocessing": ["preprocess", "augment", "normali", "预处理", "增强"],
             "baseline.metrics": ["metric", "指标"], "baseline.statistics": ["statistic", "std", "significance", "mean", "统计", "标准差", "显著"], "repro.code_version": ["code version", "commit", "tag", "release", "代码版本"],
             "repro.commands": ["command", "script", "命令", "脚本"], "repro.data_access": ["data access", "obtain", "download", "availability", "url", "获取", "下载", "数据可得"]}

QUESTION = {
    "setup.seed": "Which random seed(s) were used for training and for the data split? (none is written in the code, configs or logs)",
    "setup.repeats": "How many independent runs / folds were done, or was it a single run? Results reported as mean ± std need the count.",
    "setup.epochs": "How many epochs or steps were trained (and was the best epoch or the last one reported)?",
    "setup.batch_size": "What batch size was used for training and for evaluation?",
    "setup.optimizer": "Which optimizer (and its hyper-parameters) was used?",
    "setup.learning_rate": "What learning rate and schedule (warm-up, decay) were used?",
    "setup.hardware": "Which GPU / CPU model and how many devices ran the experiments?",
    "setup.software": "Which framework, CUDA and Python versions ran the experiments? (a requirements file or log banner would give them)",
    "data.name": "What is the dataset's name?",
    "data.version": "Which version / release of the dataset was used (a version number, date or DOI)?",
    "data.size": "How many samples are in the dataset and in each split?",
    "data.split": "What are the split proportions and how was the split drawn (random with which seed, by region / time, stratified)?",
    "data.preprocessing": "Which preprocessing and augmentation steps were applied (resizing, normalisation statistics, cropping, tokenisation)?",
    "baseline.metrics": "Which metrics were reported and what are their values for the proposed method and each baseline? (no results file or log metrics were found)",
    "baseline.statistics": "Were results averaged over runs? Please give mean ± std, the number of runs and any significance test.",
    "repro.code_version": "Which code version (git commit hash or release tag) produced the results?",
    "repro.commands": "Which commands run the experiments end to end? (no script or command line was found)",
    "repro.data_access": "Where can the data be obtained (URL, DOI, access conditions)?",
}
