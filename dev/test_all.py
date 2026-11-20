#!/usr/bin/env python3
"""Regression tests for methods-ledger (stdlib unittest). Run: python3 dev/test_all.py [-v]
Fixtures: dev/fixtures/project (an INVENTED experiment project: config, argparse code, log, README, requirements, results table) and two drafts of its methods section."""
import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUB = os.path.join(ROOT, "submission") if os.path.isdir(os.path.join(ROOT, "submission")) else ROOT
sys.path.insert(0, os.path.join(SUB, "scripts"))
import extract  # noqa: E402
import facts as F  # noqa: E402
import mini  # noqa: E402
import mw  # noqa: E402

warnings.simplefilter("ignore", ResourceWarning)
PROJ = os.path.join(HERE, "fixtures", "project")
GOOD = open(os.path.join(HERE, "fixtures", "draft_good.md"), encoding="utf-8").read()
BAD = open(os.path.join(HERE, "fixtures", "draft_bad.md"), encoding="utf-8").read()


def run(*args):
    buf = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buf):
        try:
            mw.main(list(args))
        except SystemExit as e:
            code = e.code or 0
    return code, buf.getvalue()


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def path(self, *p):
        return os.path.join(self.tmp, *p)

    def write(self, name, text):
        os.makedirs(os.path.dirname(self.path(name)), exist_ok=True)
        with open(self.path(name), "w", encoding="utf-8") as fh:
            fh.write(text)
        return self.path(name)

    def prep(self, src=PROJ):
        code, t = run("run", "--src", src, "--out", self.path("out"))
        self.assertEqual(code, 0, t)
        return self.path("out", ".work")

    def check(self, draft, work=None):
        work = work or self.prep()
        code, t = run("check", work, self.write("draft.md", draft))
        res = json.load(open(os.path.join(work, "check.json"), encoding="utf-8"))
        return work, res, {i["code"] for i in res["issues"]}


class TestMini(unittest.TestCase):
    def test_yaml(self):
        o = mini.parse_yaml("a: 1e-3\nb:\n- x: 1\n  y: [1, 2]\n- z\nc: {k: v}\nd: |\n  l1\n  l2\ne: 'q'  # c\nf: none\n")
        self.assertEqual(o, {"a": 0.001, "b": [{"x": 1, "y": [1, 2]}, "z"], "c": {"k": "v"}, "d": "l1\nl2", "e": "q", "f": "none"})

    def test_yaml_sequence_at_key_indent_and_flow_on_next_line(self):
        o = mini.parse_yaml("jobs:\n- job: a\n  steps:\n  - run: x\nback:\n  [\n    [-1, 1, Conv],  # c\n    [2, 3, C3]\n  ]\n")
        self.assertEqual(o["jobs"][0]["steps"][0]["run"], "x")
        self.assertEqual(o["back"], [[-1, 1, "Conv"], [2, 3, "C3"]])

    def test_toml(self):
        o = mini.parse_toml('[train]\nlr = 3e-4 # x\nlayers=[1,\n 2]\n[[opt]]\nname="a"\n[[opt]]\nname="b"\n[x."y.z"]\nk = { a = 1, b = [1, 2] }\n')
        self.assertEqual(o["train"], {"lr": 0.0003, "layers": [1, 2]})
        self.assertEqual([x["name"] for x in o["opt"]], ["a", "b"])
        self.assertEqual(o["x"]["y.z"]["k"], {"a": 1, "b": [1, 2]})

    def test_flatten_and_lines(self):
        pl = {}
        o = mini.parse_yaml("a:\n  b: 1\n  c: [1, 2]\n", pl)
        self.assertEqual(mini.flatten(o), [("a.b", 1), ("a.c", [1, 2])])
        self.assertEqual(pl["a.b"], 2)


class TestExtract(unittest.TestCase):
    def py(self, code):
        facts, note = extract.from_python(code, "t.py")
        self.assertEqual(note, "")
        return {(f["name"], json.dumps(f["value"])): f for f in facts}

    def test_argparse_calls_assign_dataclass(self):
        f = self.py("import argparse\np=argparse.ArgumentParser()\np.add_argument('--lr', '--learning-rate', type=float, default=3e-4)\np.add_argument('--amp', action='store_true')\n"
                    "torch.manual_seed(7)\nopt = AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)\nBATCH_SIZE = 64\ncfg = dict(optimizer=dict(type='SGD', lr=0.02))\n"
                    "from dataclasses import dataclass\n@dataclass\nclass TrainConfig:\n    epochs: int = 12\n    dropout: float = 0.1\n")
        for key in (("lr", "0.0003"), ("seed", "7"), ("lr", "0.001"), ("weight_decay", "0.05"), ("BATCH_SIZE", "64"), ("cfg.optimizer.lr", "0.02"), ("epochs", "12"), ("dropout", "0.1")):
            self.assertIn(key, f, sorted(f))
        self.assertEqual(f[("lr", "0.0003")]["canon"], "learning_rate")
        self.assertEqual(f[("seed", "7")]["canon"], "seed")

    def test_log_patterns(self):
        log = ("INFO Namespace(epochs=50, lr=0.001, seed=42)\nPyTorch 2.1.0, CUDA 12.1, Python 3.10.12\nGPU: NVIDIA A100-SXM4-40GB x 4\ntrain: 8000 samples, val: 1000 samples\n"
               "Epoch 1/50 - loss: 0.9 - val acc: 0.61 - lr: 0.001\nEpoch 50/50 - loss: 0.2 - val acc: 0.88 - lr: 0.00001\nTest accuracy: 0.871\n")
        facts, results, cmds = extract.from_log(log, "l.log")
        names = {(f["name"], str(f["value"])) for f in facts}
        for k in (("epochs", "50"), ("lr", "0.001"), ("seed", "42"), ("env.pytorch", "2.1.0"), ("env.cuda", "12.1"), ("env.python", "3.10.12"), ("hardware.gpu", "NVIDIA A100-SXM4-40GB"), ("hardware.num_gpus", "4"), ("dataset.size.train", "8000.0"), ("dataset.size.val", "1000.0")):
            self.assertIn(k, names, sorted(names))
        self.assertIn(("val_acc", 0.88), {(r["name"], r["value"]) for r in results})
        self.assertIn(("test_accuracy", 0.871), {(r["name"], r["value"]) for r in results})
        self.assertTrue(any(f["name"] == "learning_rate_last" or f["name"].endswith("_last") for f in facts))

    def test_prose_and_requirements(self):
        facts = extract.from_prose("We train with a learning rate of 3e-4, a batch size of 128 for 30 epochs on 2 NVIDIA V100 GPUs. The data has 12,000 images split 70:15:15, seeds 1, 2, 3.", "n.md")
        got = {(f["canon"] or f["name"], str(f["value"])) for f in facts}
        for k in (("learning_rate", "0.0003"), ("batch_size", "128.0"), ("epochs", "30.0"), ("num_gpus", "2"), ("dataset.size.image", "12000.0"), ("dataset.split_ratio", "70:15:15"), ("seed", "1"), ("seed", "3")):
            self.assertTrue(any(k[0] == a and k[1] in b for a, b in got), (k, sorted(got)))
        env = extract.from_requirements("torch==2.1.0\nnumpy>=1.2 # c\nFROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime\n", "requirements.txt")
        self.assertEqual({(f["name"], f["value"]) for f in env if f["name"] in ("env.torch", "env.numpy")}, {("env.torch", "2.1.0"), ("env.numpy", ">=1.2")})

    def test_values_are_never_invented(self):
        facts, note = extract.from_python("x = compute()\nlr = args.lr\nopt = Adam(model.parameters())\n", "t.py")
        self.assertEqual([f for f in facts if f["canon"] == "learning_rate"], [])

    def test_commands(self):
        facts, cmds = extract.from_commands("# c\npython train.py --lr 0.001 --epochs=5 --amp\n$ torchrun --nproc_per_node 4 train.py --batch-size 64\n", "run.sh")
        self.assertEqual(len(cmds), 2)
        self.assertEqual({(f["canon"], f["value"]) for f in facts if f["canon"]}, {("learning_rate", 0.001), ("epochs", 5), ("batch_size", 64)})


class TestConflicts(unittest.TestCase):
    def facts(self, rows):
        out = []
        for i, (name, v, kind, scope) in enumerate(rows):
            f = extract.fact(name, v, "f%d" % i, i + 1, kind, scope=scope)
            out.append(f)
        extract.assign_ids(out)
        return out

    def test_runtime_settles_defaults(self):
        fs = self.facts([("lr", 0.001, "config", None), ("lr", 0.0003, "argparse", None), ("lr", 0.001, "log", None)])
        self.assertEqual(F.conflicts(fs), [])
        t = F.param_table(fs, [], [])
        self.assertTrue(t[0]["status"].startswith("settled") or t[0]["status"] == "consistent")

    def test_config_vs_hardcoded_is_a_conflict_but_argparse_default_is_not(self):
        fs = self.facts([("weight_decay", 0.01, "config", None), ("weight_decay", 0.05, "call", None)])
        self.assertEqual(len(F.conflicts(fs)), 1)
        fs = self.facts([("weight_decay", 0.01, "config", None), ("weight_decay", 0.05, "argparse", None)])
        self.assertEqual(F.conflicts(fs), [])

    def test_runtime_disagreement_and_scopes(self):
        fs = self.facts([("lr", 0.001, "log", None), ("lr", 0.01, "cli", None)])
        self.assertEqual(len(F.conflicts(fs)), 1)
        fs = self.facts([("train.batch_size", 32, "config", None), ("eval.batch_size", 64, "config", None)])
        self.assertEqual(F.conflicts(fs), [])

    def test_equal_scheduler_names(self):
        fs = self.facts([("scheduler", "cosine", "config", None), ("scheduler", "CosineAnnealingLR", "call", None)])
        self.assertEqual(F.conflicts(fs), [])


class TestRunAndBrief(Tmp):
    def test_brief_on_fixture(self):
        code, t = run("run", "--src", PROJ, "--out", self.path("out"))
        self.assertEqual(code, 0)
        self.assertIn("batch_size = 32 (config.yaml:11) | 16 (train.py:20) [train] CONFLICT", t)
        self.assertIn("learning_rate = 0.001 [train] settled by run log", t)
        self.assertIn("hardware.gpu = NVIDIA A100-SXM4-40GB", t)
        self.assertIn("dataset.size.train = 8000", t)
        self.assertIn("setup.repeats", t)
        self.assertIn("python train.py --config config.yaml --epochs 50", t)

    def test_every_fact_has_a_source(self):
        work = self.prep()
        M = json.load(open(os.path.join(work, "materials.json"), encoding="utf-8"))
        self.assertTrue(M["facts"])
        self.assertTrue(all(f["source"] and f["id"] for f in M["facts"]))

    def test_existing_draft_is_not_evidence_for_itself(self):
        shutil.copytree(PROJ, self.path("p"))
        self.write("p/method_draft.md", "## Method\n\nWe trained with a learning rate of 0.5 for 999 epochs.\n\n## Experiments\n\n" + "Sentence one is here. " * 9)
        code, t = run("run", "--src", self.path("p"), "--out", self.path("o"))
        self.assertIn("Existing draft", t)
        self.assertIn("value_mismatch", t)
        M = json.load(open(self.path("o", ".work", "materials.json"), encoding="utf-8"))
        self.assertFalse([f for f in M["facts"] if f["value"] == 999])

    def test_case_inline_text_and_settings(self):
        case = {"case_id": "c", "task": "写一段中文的方法与实验章节，约300字。", "notes": "我们使用学习率 0.0005，批大小为 128，训练 20 个 epoch。数据集包含 5000 张图像。" + "补充说明。" * 30, "settings": {"optimizer": "SGD", "momentum": 0.9, "lr": 0.0005}}
        os.makedirs(self.path("in"))
        cp = self.write("in/case.json", json.dumps(case, ensure_ascii=False))
        code, t = run("run", "--case", cp, "--out", self.path("o"))
        self.assertEqual(code, 0, t)
        M = json.load(open(self.path("o", ".work", "materials.json"), encoding="utf-8"))
        names = {(f["canon"] or f["name"], str(f["value"])) for f in M["facts"]}
        self.assertIn(("momentum", "0.9"), names)
        self.assertTrue(any(c == "batch_size" for c, _ in names))
        self.assertEqual(M["lang"], "zh")
        self.assertEqual(M["task"]["length"], 300)


class TestCheck(Tmp):
    def test_good_draft_has_no_errors_and_every_number_is_tied(self):
        work, res, codes = self.check(GOOD)
        self.assertEqual([i for i in res["issues"] if i["severity"] == "error"], [])
        self.assertEqual(res["stats"]["numbers"], res["stats"]["sourced"])

    def test_seeded_defects_are_found(self):
        work, res, codes = self.check(BAD)
        for want in ("value_mismatch", "unsourced_number", "condition_mixup", "placeholder_left", "generic_limitation", "unverified_comparability", "gap_not_marked", "open_conflict"):
            self.assertIn(want, codes, codes)
        msgs = " | ".join(i["message"] for i in res["issues"])
        for frag in ("epochs 100", "learning rate 0.01", "seed 7", "PyTorch 2.0", "Python 3.8", "optimizer is Adam but", "GPU is NVIDIA V100", "dataset version 2.0", "image size 224", "12,000", "7:2:1", "0.85"):
            self.assertIn(frag, msgs, frag)

    def test_condition_mixup_names_the_other_stage(self):
        work, res, codes = self.check("## Setup\n\nDuring evaluation we use a batch size of 32.\n")
        self.assertIn("condition_mixup", codes)

    def test_derived_numbers_are_accepted(self):
        work, res, codes = self.check("## Dataset\n\nThe 10,000 tiles split into 80% / 10% / 10% (8,000, 1,000 and 1,000 tiles).\n")
        self.assertNotIn("unsourced_number", codes, [i["message"] for i in res["issues"]])

    def test_percent_and_scientific_forms_match(self):
        work, res, codes = self.check("## Setup\n\nThe learning rate was 1 x 10^-3 and the test accuracy was 87.1%.\n")
        self.assertNotIn("unsourced_number", codes)
        self.assertNotIn("value_mismatch", codes)

    def test_confirm_markers_are_open_items_not_errors(self):
        work, res, codes = self.check("## Setup\n\nThe seed was 42 [[CONFIRM: number of repeated runs]].\n")
        self.assertIn("needs_author_confirmation", codes)
        self.assertEqual([i for i in res["issues"] if i["severity"] == "error"], [])

    def test_limitation_with_scenario_is_accepted(self):
        work, res, codes = self.check("## Limitations\n\nThe model was only tested on Sentinel imagery; when the target region lies outside the 14 training regions it may not generalize.\n")
        self.assertNotIn("generic_limitation", codes)

    def test_chinese_draft(self):
        work, res, codes = self.check("## 实验设置\n\n模型训练 100 个 epoch，学习率为 0.01，批大小为 32。\n")
        msgs = " | ".join(i["message"] for i in res["issues"])
        self.assertIn("epochs 100", msgs)
        self.assertIn("learning rate 0.01", msgs)

    def test_check_is_deterministic(self):
        work, r1, _ = self.check(BAD)
        run("check", work, self.path("draft.md"))
        r2 = json.load(open(os.path.join(work, "check.json"), encoding="utf-8"))
        self.assertEqual(r1["issues"], r2["issues"])


class TestIngest(Tmp):
    def test_resolve_needs_an_observed_value_and_a_reason(self):
        work = self.prep()
        code, t = run("ingest", work, self.write("b.txt", "=== resolve batch_size train\nvalue: 77\nreason: x\n=== resolve batch_size train\nvalue: 32\n=== resolve nothing_here\nvalue: 1\nreason: y\n"))
        self.assertEqual(code, 2)
        self.assertEqual(t.count("REJECTED"), 2)
        code, t = run("ingest", work, self.write("b.txt", "=== resolve batch_size train\nvalue: 32\nreason: config.yaml is the file the training command loads\n"))
        self.assertEqual(code, 0, t)
        code, t = run("check", work, self.write("d.md", "## Setup\n\nThe batch size was 32.\n"))
        res = json.load(open(os.path.join(work, "check.json"), encoding="utf-8"))
        self.assertNotIn("open_conflict", {i["code"] for i in res["issues"] if i.get("canon") == "batch_size"})
        self.assertNotIn("conflicted_value_used", {i["code"] for i in res["issues"]})

    def test_confirm_needs_a_verbatim_quote_from_the_materials(self):
        work = self.prep()
        code, t = run("ingest", work, self.write("b.txt", "=== confirm setup.repeats\nvalue: 5\nquote: we repeated every run five times with different seeds\n"))
        self.assertEqual(code, 2)
        self.assertIn("REJECTED", t)

    def test_confirm_with_a_quote_from_the_task_text(self):
        os.makedirs(self.path("in"))
        self.write("in/task.md", "Write the section. The author says: every experiment was repeated 5 times with seeds 1 to 5.\n")
        shutil.copytree(PROJ, self.path("in", "proj"))
        cp = self.write("in/case.json", json.dumps({"case_id": "c"}))
        code, t = run("run", "--case", cp, "--out", self.path("o"))
        work = self.path("o", ".work")
        code, t = run("ingest", work, self.write("b.txt", "=== confirm repeats\nvalue: 5\nquote: every experiment was repeated 5 times\n"))
        self.assertEqual(code, 0, t)
        work, res, codes = self.check("## Setup\n\nEvery experiment was repeated 5 times.\n", work)
        self.assertNotIn("unsourced_number", codes)

    def test_limitation_needs_a_scenario(self):
        work = self.prep()
        code, t = run("ingest", work, self.write("b.txt", "=== limitation L1\nscenario: bad\n"))
        self.assertEqual(code, 2)


class TestBuild(Tmp):
    def build(self, draft=GOOD, work=None, *extra):
        work = work or self.prep()
        code, t = run("build", work, self.write("draft.md", draft), "--out", self.path("out"), *extra)
        return work, code, t

    def test_outputs(self):
        work, code, t = self.build()
        out = self.path("out")
        for f in ("section.md", "section.docx", "section.tex", "parameters.csv", "parameters.md", "datasets_environment.csv", "commands.md", "source_map.csv", "source_map.json", "questions.md", "review_checklist.md", "result.json"):
            self.assertTrue(os.path.isfile(os.path.join(out, f)), f)
        import zipfile
        import xml.dom.minidom as md
        md.parseString(zipfile.ZipFile(os.path.join(out, "section.docx")).read("word/document.xml"))
        q = open(os.path.join(out, "questions.md"), encoding="utf-8").read()
        self.assertIn("batch_size", q)
        self.assertIn("[missing]", q)
        res = json.load(open(os.path.join(out, "result.json"), encoding="utf-8"))
        self.assertEqual(res["status"], "partial")  # open conflicts
        rows = open(os.path.join(out, "parameters.csv"), encoding="utf-8").read()
        self.assertIn("CONFLICT", rows)

    def test_status_success_when_everything_is_settled(self):
        work = self.prep()
        res = ("=== resolve batch_size train\nvalue: 32\nreason: config.yaml is loaded by the command in run.sh\n=== resolve weight_decay train\nvalue: 0.01\nreason: the config\n=== resolve dropout train\nvalue: 0.2\nreason: the config\n")
        code, t = run("ingest", work, self.write("r.txt", res))
        self.assertEqual(code, 0, t)
        work, code, t = self.build(GOOD, work)
        self.assertEqual(json.load(open(self.path("out", "result.json"), encoding="utf-8"))["status"], "success", t)

    def test_contract_fields(self):
        os.makedirs(self.path("in"))
        shutil.copytree(PROJ, self.path("in", "proj"))
        cp = self.write("in/case.json", json.dumps({"case_id": "c1", "output_contract": {"fields": ["case_id", "目标产物", "事实与来源映射", "版本与边界", "reviewer_score"]}}))
        run("run", "--case", cp, "--out", self.path("out"))
        work = self.path("out", ".work")
        code, t = run("build", work, self.write("draft.md", GOOD), "--out", self.path("out"))
        res = json.load(open(self.path("out", "result.json"), encoding="utf-8"))
        self.assertEqual(res["case_id"], "c1")
        self.assertIn("text", res["目标产物"])
        self.assertIn("numbers", res["事实与来源映射"])
        self.assertIn("questions_for_author", res["版本与边界"])
        self.assertIsNone(res["reviewer_score"])
        self.assertEqual(code, 2)
        ans = self.write("a.json", json.dumps({"reviewer_score": "set by the platform"}))
        code, t = run("build", work, self.path("draft.md"), "--out", self.path("out"), "--answers", ans)
        self.assertEqual(code, 0)

    def test_skeleton_is_fact_bound_and_marks_gaps(self):
        work = self.prep()
        code, t = run("skeleton", work)
        self.assertIn("50 epochs", t)
        self.assertIn("[[CONFIRM:", t)
        self.assertIn("LS-Tiles", t)
        self.assertNotIn("None", t)
        work, res, codes = self.check(t, work)
        self.assertEqual([i for i in res["issues"] if i["severity"] == "error"], [], [i["message"] for i in res["issues"] if i["severity"] == "error"])

    def test_no_materials_is_reported_not_crashed(self):
        os.makedirs(self.path("empty"))
        code, t = run("run", "--src", self.path("empty"), "--out", self.path("out"))
        self.assertEqual(code, 0, t)
        work, code, t = self.build("## Setup\n\nWe trained for 10 epochs.\n", self.path("out", ".work"))
        res = json.load(open(self.path("out", "result.json"), encoding="utf-8"))
        self.assertEqual(res["status"], "partial")
        self.assertTrue(any(u["kind"] == "no_facts" for u in res["boundaries"]["uncertainty"]))

    def test_unreadable_files_do_not_stop_the_run(self):
        shutil.copytree(PROJ, self.path("p"))
        self.write("p/bad.yaml", "a: [1, 2\nb: {")
        self.write("p/broken.py", "def (:\n")
        code, t = run("run", "--src", self.path("p"), "--out", self.path("o"))
        self.assertEqual(code, 0, t)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
