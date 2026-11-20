#!/usr/bin/env python3
"""extract.py - what the experiment's own files say: parameters, datasets, environment, commands, results, with the file and line each value comes from.

Sources: configuration files (YAML / JSON / TOML / INI / .env), Python code and notebooks (argparse defaults, call keywords, constants, dataclass fields; AST), shell scripts / Makefiles / READMEs (commands),
logs (key: value lines, Namespace(...) dumps, JSON lines, metric lines, environment banners), CSV result tables, requirement / environment / Docker files, and prose notes (sentences that name a parameter and a value).
Every fact: {id, name, canon, value, num, scope, kind, file, line, context}. `canon` is the canonical parameter (learning_rate, batch_size, seed ...) when the name is a known alias.
Nothing is guessed: a value that is not written in a file is not a fact.
"""
from __future__ import annotations

import ast
import csv
import io
import json
import os
import re
from collections import defaultdict

import mini
import nlib

# ------------------------------------------------------------------------------------------------ canonical parameters
# canon -> (regex over a parameter NAME (last path part, lower-case), regex over prose that names it, value type)
PARAMS = {
    "learning_rate": (r"^(?:base_?|init(?:ial)?_?|max_?|peak_?|min_?)?(?:lr|learning[_-]?rate)(?:_?(?:init|max|min|base))?$", r"learning[ -]rate|\blr\b|学习率", "num"),
    "batch_size": (r"^(?:train_?|eval_?|test_?|val_?|per_device_(?:train|eval)_?|global_?|micro_?)?(?:batch[_-]?size|bs|bsz|batch)$", r"batch[ -]size|mini-?batch(?: size)?|batches of|批大小|批量大小|批尺寸", "num"),
    "epochs": (r"^(?:num_?|n_?|max_?|total_?)?(?:epochs?|train_epochs|num_train_epochs)$", r"(?:number of |total |max(?:imum)? )?epochs?|训练轮数|轮次|个epoch", "num"),
    "seed": (r"^(?:random_?|manual_?|global_?|rng_?)?(?:seed|random_state|seeds)$", r"random seeds?|\bseeds?\b|随机种子|种子", "num"),
    "weight_decay": (r"^(?:weight_?decay|wd)$", r"weight[ -]decay|权重衰减", "num"),
    "momentum": (r"^momentum$", r"momentum|动量", "num"),
    "dropout": (r"^(?:p_?)?drop(?:out)?(?:_?(?:rate|prob|p|ratio))?$", r"dropout(?: (?:rate|probability|ratio))?|丢弃率", "num"),
    "optimizer": (r"^(?:optim(?:izer)?|opt)(?:_?(?:type|name|class))?$", r"optimi[sz]er|优化器", "str"),
    "scheduler": (r"^(?:lr_?)?(?:sched(?:uler)?)(?:_?(?:type|name))?$", r"(?:lr |learning[ -]rate )?schedul(?:er|e)|学习率调度", "str"),
    "warmup": (r"^(?:warmup|warm_up)(?:_?(?:steps|epochs|ratio|iters|iterations))?$", r"warm-?up(?: (?:steps|epochs|ratio|iterations))?|预热", "num"),
    "gamma": (r"^gamma$", r"\bgamma\b", "num"),
    "loss": (r"^(?:loss|criterion)(?:_?(?:fn|type|name|function))?$", r"loss function|criterion|损失函数", "str"),
    "model": (r"^(?:model|arch|architecture|backbone|net|network)(?:_?(?:name|type))?$", r"backbone|architecture|骨干网络", "str"),
    "hidden_size": (r"^(?:hidden_?(?:size|dim|dims|units)|d_model|embed(?:ding)?_?dim)$", r"hidden (?:size|dimension|units)|embedding (?:size|dimension)|隐藏层?(?:维度|大小)", "num"),
    "num_layers": (r"^(?:num_?|n_?)?layers?$|^n_?layer$|^num_hidden_layers$", r"(?:number of )?layers", "num"),
    "num_heads": (r"^(?:num_?|n_?)?(?:attention_?)?heads$|^num_attention_heads$", r"(?:number of )?attention heads|heads", "num"),
    "max_length": (r"^(?:max_?)?(?:seq(?:uence)?_?len(?:gth)?|length|len|max_seq_length|max_position_embeddings|context_length)$", r"(?:maximum |max )?(?:sequence|context|input) length|最大序列长度", "num"),
    "image_size": (r"^(?:img|image|input|crop|resize)_?(?:size|res(?:olution)?|shape)$|^resolution$", r"image (?:size|resolution)|input (?:size|resolution)|resized? to|crop(?:ped)? to|分辨率", "num"),
    "test_size": (r"^(?:test|val(?:idation)?|dev)_?(?:size|ratio|split|frac(?:tion)?)$|^(?:train)_?(?:ratio|split|frac(?:tion)?)$", r"(?:test|validation|val|training|train) (?:size|ratio|split|fraction)|split ratio|划分比例", "num"),
    "num_workers": (r"^(?:num_?|n_?)?workers$", r"data loader workers|workers", "num"),
    "grad_clip": (r"^(?:grad(?:ient)?_?clip(?:_?norm)?|max_grad_norm|clip_grad_norm|clip_?norm)$", r"gradient clipping|grad(?:ient)? clip|max grad norm|梯度裁剪", "num"),
    "patience": (r"^(?:early_?stop(?:ping)?_?)?patience$", r"patience|early stopping", "num"),
    "precision": (r"^(?:precision|fp16|bf16|amp|mixed_?precision|use_amp)$", r"mixed[ -]precision|fp16|bf16|amp|混合精度", "any"),
    "num_gpus": (r"^(?:num_?|n_?)?gpus?$|^world_size$|^devices$", r"gpus?|显卡", "num"),
    "temperature": (r"^temperature$|^tau$", r"temperature|温度系数", "num"),
    "repeats": (r"^(?:num_?|n_?)?(?:runs|repeats?|restarts?|trials|folds?|k_?fold)$", r"(?:independent |repeated )?runs|repeats|restarts|trials|folds?|重复实验|次重复", "num"),
    "beta": (r"^betas?$|^beta[12]$", r"betas?|\bβ", "num"),
    "eps": (r"^(?:eps|epsilon)$", r"epsilon|eps", "num"),
    "steps": (r"^(?:max_?|total_?|num_?)?(?:steps|iters|iterations|train_steps|max_iters)$", r"(?:training |total )?(?:steps|iterations)|迭代次数", "num"),
}
PARAM_RX = {k: re.compile(v[0], re.I) for k, v in PARAMS.items()}
PHRASE_RX = {k: re.compile(r"(?<![A-Za-z])(?:%s)(?![A-Za-z])" % v[1], re.I) for k, v in PARAMS.items()}
OPTIMIZERS = "AdamW|Adam|SGD|RMSprop|Adagrad|Adadelta|Adamax|NAdam|RAdam|LAMB|LARS|Lion|Adafactor|AdamP|Ranger|Muon"
OPT_RX = re.compile(r"\b(%s)\b" % OPTIMIZERS)
SCHEDS = "StepLR|MultiStepLR|CosineAnnealingLR|CosineAnnealingWarmRestarts|ExponentialLR|ReduceLROnPlateau|OneCycleLR|LinearLR|PolynomialLR|cosine|linear|constant|step|polynomial|exponential|warmup_cosine|inverse_sqrt"
ML_CALLEE = re.compile(r"optim|sched|loader|dataset|model|net|layer|conv|linear|dropout|trainer|split|seed|norm|resize|crop|flip|augment|transform|embed|lstm|gru|attention|transformer|loss|clip|warmup|adam|sgd|rms|batch|encoder|decoder|pool|train|fit|config|args|params|init", re.I)
PKGS = r"torch|pytorch|torchvision|torchaudio|tensorflow|keras|jax|flax|numpy|scipy|pandas|scikit-learn|sklearn|transformers|datasets|accelerate|lightning|pytorch-lightning|cuda|cudnn|nccl|python|opencv|opencv-python|timm|xgboost|lightgbm|catboost|mmcv|mmdet|mmengine|detectron2|onnx|onnxruntime|deepspeed|peft|diffusers|tokenizers|matplotlib|seaborn|statsmodels|networkx|geopandas|rasterio|gdal|xarray|netcdf4|h5py|zarr|dask"


def canon_of(name):
    if name == "hardware.num_gpus":
        return "num_gpus"
    if name.startswith(("hardware.", "env.", "dataset.", "table_rows.")):
        return ""
    n = re.sub(r"\[\d+\]", "", name).split(".")[-1].lower().strip("-_ ")
    n = n.lstrip("-").replace("-", "_")
    for k, rx in PARAM_RX.items():
        if rx.match(n):
            return k
    return ""


def scope_of(path, canon="", callee=""):
    t = (path + " " + callee).lower()
    if re.search(r"(?<![a-z])(test|testing|eval|evaluation|inference)(?![a-z])|测试|评测|评估|推理", t):
        return "test"
    if re.search(r"(?<![a-z])(val|valid|validation|dev)(?![a-z])|验证", t):
        return "val"
    if re.search(r"(?<![a-z])(train|training|fit|finetune|fine_tune|pretrain)(?![a-z])|训练|微调", t):
        return "train"
    if re.search(r"(?<![a-z])(data|dataset|dataloader|transform|augment)(?![a-z])", t):
        return "data"
    if re.search(r"(?<![a-z])(model|net|network|backbone|encoder|decoder)(?![a-z])", t):
        return "model"
    if re.search(r"(?<![a-z])(optim|optimizer|sched|scheduler|lr)(?![a-z])", t):
        return "optim"
    return "general"


def to_num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", "").replace("_", "")
        s = re.sub(r"\s*[×x*]\s*10\s*\^?\s*\{?\s*([-−–+]?\d+)\}?", r"e\1", s)
        s = s.replace("−", "-").replace("–", "-")
        if re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?%?", s):
            try:
                return float(s.rstrip("%")) if not s.endswith("%") else float(s[:-1])
            except ValueError:
                return None
    return None


def fact(name, value, file, line, kind, context="", callee="", scope=None, unit=""):
    canon = canon_of(name)
    sc = scope or scope_of(name, canon, callee)
    if sc == "general" and kind in ("call", "assign", "argparse") and not scope:
        base = os.path.basename(file).lower()
        if re.search(r"eval|test|infer|predict", base):
            sc = "test"
        elif re.search(r"train|fit|finetune|pretrain", base):
            sc = "train"
    return {"name": name, "canon": canon, "value": value, "num": to_num(value) if not isinstance(value, (list, dict)) else None, "scope": sc, "kind": kind, "file": file, "line": line, "context": context[:200], "unit": unit, "callee": callee}


# ------------------------------------------------------------------------------------------------ configuration files
def from_config(path, text, rel):
    facts = []
    try:
        obj, pl = mini.load_any(path, text)
    except Exception as e:
        return facts, "unreadable config (%s: %s)" % (type(e).__name__, str(e)[:60])
    for p, v in mini.flatten(obj):
        if v is None or isinstance(v, dict):
            continue
        line = pl.get(p) or pl.get(re.sub(r"\[\d+\]", "", p))
        if line is None and not pl:
            line = find_line(text, p.split(".")[-1])
        f = fact(p, v, rel, line, "config")
        facts.append(f)
        if isinstance(v, list) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            for i, x in enumerate(v):
                g = fact("%s[%d]" % (p, i), x, rel, line, "config")
                g["canon"] = f["canon"]
                g["scope"] = f["scope"]
                facts.append(g)
    return facts, ""


def find_line(text, key):
    for i, ln in enumerate(text.splitlines(), 1):
        if re.search(r"[\"']?%s[\"']?\s*[:=]" % re.escape(key), ln):
            return i
    return None


# ------------------------------------------------------------------------------------------------ Python code
def lit(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = lit(node.operand)
            return -v if isinstance(v, (int, float)) else None
        if isinstance(node, ast.Attribute):
            return None
        if isinstance(node, ast.Name) and node.id in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[node.id]
        return None


def dotted(node):
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + "." + node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return dotted(node.func)
    return ""


def from_python(text, rel):
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return [], "not parseable as Python (line %s)" % e.lineno
    facts = []
    consts = {}

    def add(name, value, line, kind, callee="", ctx=""):
        if value is None or isinstance(value, (dict, set)):
            return
        facts.append(fact(name, value, rel, line, kind, ctx or callee, callee))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = dotted(node.func)
            last = fn.split(".")[-1]
            if last == "add_argument":
                names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                long = next((n for n in names if n.startswith("--")), names[0] if names else "")
                kw = {k.arg: k.value for k in node.keywords if k.arg}
                if "default" in kw and long:
                    v = lit(kw["default"])
                    if v is None and isinstance(kw["default"], ast.Name) and kw["default"].id in consts:
                        v = consts[kw["default"].id]
                    h = lit(kw["help"]) if "help" in kw else ""
                    add(long.lstrip("-"), v, node.lineno, "argparse", "", ("--%s default; %s" % (long.lstrip("-"), h or "")).strip())
                elif long and kw.get("action") is not None and lit(kw["action"]) in ("store_true", "store_false"):
                    add(long.lstrip("-"), lit(kw["action"]) == "store_false", node.lineno, "argparse", "", "--%s flag (default %s)" % (long.lstrip("-"), lit(kw["action"]) == "store_false"))
                continue
            seedlike = {"manual_seed", "seed", "set_seed", "seed_everything", "RandomState", "default_rng", "random_seed", "set_random_seed", "PRNGKey"}
            if last in seedlike and node.args:
                v = lit(node.args[0])
                if isinstance(v, int):
                    add("seed", v, node.lineno, "call", fn, "%s(%s)" % (fn, v))
                continue
            if last in ("Dropout", "Dropout2d", "AlphaDropout") and node.args:
                add("dropout", lit(node.args[0]), node.lineno, "call", fn, "%s(%s)" % (fn, lit(node.args[0])))
            if OPT_RX.fullmatch(last) or last in ("Adam", "SGD", "AdamW"):
                facts.append(fact("optimizer", last, rel, node.lineno, "call", "%s(...)" % fn, fn))
            if re.fullmatch(SCHEDS, last, re.I) or last.endswith("LR"):
                facts.append(fact("scheduler", last, rel, node.lineno, "call", "%s(...)" % fn, fn))
            if last == "train_test_split" or ML_CALLEE.search(fn):
                for k in node.keywords:
                    if not k.arg:
                        continue
                    v = lit(k.value)
                    if v is None or isinstance(v, (dict, set)) or (isinstance(v, str) and len(v) > 80):
                        continue
                    if isinstance(v, (list, tuple)) and not all(isinstance(x, (int, float, str)) for x in v):
                        continue
                    if canon_of(k.arg) or (isinstance(v, (int, float)) and not isinstance(v, bool)) or k.arg in ("shuffle", "pretrained", "augment", "normalize"):
                        add(k.arg, list(v) if isinstance(v, tuple) else v, getattr(k.value, "lineno", node.lineno), "call", fn, "%s(%s=%s)" % (fn, k.arg, v))
                if last in ("Adam", "AdamW", "SGD", "RMSprop", "Adagrad") and node.args and len(node.args) >= 2:
                    pass
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            val = node.value
            if val is None:
                continue
            for t in targets:
                if isinstance(t, ast.Name):
                    name = t.id
                elif isinstance(t, ast.Attribute):
                    name = t.attr
                else:
                    continue
                v = lit(val)
                if isinstance(v, (int, float, str, bool)) and not isinstance(v, type(None)):
                    consts[name] = v
                    if canon_of(name) or (name.isupper() and isinstance(v, (int, float)) and not isinstance(v, bool)) or (isinstance(v, str) and name.isupper() and re.search(r"path|dir|name|dataset|model", name, re.I)):
                        add(name, v, node.lineno, "assign", "", "%s = %s" % (name, v))
                elif isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v) and (canon_of(name) or name.isupper()):
                    add(name, list(v), node.lineno, "assign", "", "%s = %s" % (name, list(v)))
                elif isinstance(val, ast.Dict) or (isinstance(val, ast.Call) and dotted(val.func) in ("dict", "Config", "ConfigDict", "EasyDict", "SimpleNamespace", "Namespace")):
                    for p, x in flatten_dictish(val, name):
                        add(p, x, getattr(val, "lineno", node.lineno), "assign", "", "%s = %s" % (p, x))
        elif isinstance(node, ast.ClassDef) and re.search(r"config|args|param|hparam|setting|option", node.name, re.I):
            for b in node.body:
                if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name) and b.value is not None:
                    v = lit(b.value)
                    if v is None and isinstance(b.value, ast.Call) and dotted(b.value.func).endswith("field"):
                        for k in b.value.keywords:
                            if k.arg == "default":
                                v = lit(k.value)
                    if isinstance(v, (int, float, str, bool)):
                        add(b.target.id, v, b.lineno, "assign", node.name, "%s.%s = %s" % (node.name, b.target.id, v))
                elif isinstance(b, ast.Assign) and len(b.targets) == 1 and isinstance(b.targets[0], ast.Name):
                    v = lit(b.value)
                    if isinstance(v, (int, float, str, bool)):
                        add(b.targets[0].id, v, b.lineno, "assign", node.name, "%s.%s = %s" % (node.name, b.targets[0].id, v))
    return dedup(facts), ""


def flatten_dictish(node, prefix):
    out = []
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            kk = lit(k) if k is not None else None
            if kk is None:
                continue
            out += flatten_dictish(v, "%s.%s" % (prefix, kk))
        return out
    if isinstance(node, ast.Call) and dotted(node.func) in ("dict", "Config", "ConfigDict", "EasyDict", "SimpleNamespace", "Namespace"):
        for k in node.keywords:
            if k.arg:
                out += flatten_dictish(k.value, "%s.%s" % (prefix, k.arg))
        return out
    v = lit(node)
    if isinstance(v, tuple):
        v = list(v)
    if isinstance(v, (int, float, str, bool, list)) and v is not None:
        if isinstance(v, str) and len(v) > 80:
            return []
        out.append((prefix, v))
    return out


def dedup(facts):
    seen, out = set(), []
    for f in facts:
        k = (f["file"], f["line"], f["name"], json.dumps(f["value"], default=str))
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


def from_notebook(text, rel):
    try:
        nb = json.loads(text)
    except ValueError:
        return [], [], "not a valid notebook"
    code, outs = [], []
    for c in nb.get("cells", []):
        src = "".join(c.get("source", []))
        if c.get("cell_type") == "code":
            code.append(src)
            for o in c.get("outputs", []):
                t = o.get("text") or (o.get("data", {}).get("text/plain") if isinstance(o.get("data"), dict) else "")
                if t:
                    outs.append("".join(t))
    facts, note = from_python("\n\n".join(code), rel)
    return facts, "\n".join(outs), note


# ------------------------------------------------------------------------------------------------ commands, logs, prose
CMD = re.compile(r"^\s*(?:\$|>|%|#!\s*)?\s*((?:CUDA_VISIBLE_DEVICES=\S+\s+|[A-Z_]+=\S+\s+)*(?:python3?|torchrun|accelerate launch|deepspeed|bash|sh|srun|sbatch|mpirun|horovodrun|make|jupyter nbconvert|python -m \S+)\s+[^\n]*)$")
FLAG = re.compile(r"(?:^|\s)--?([A-Za-z][\w\-.]*)(?:[= ]+(?!-)([^\s'\"]+|\"[^\"]*\"|'[^']*'))?")


def from_commands(text, rel, kind="cli"):
    facts, cmds = [], []
    for no, ln in enumerate(text.splitlines(), 1):
        m = CMD.match(ln.rstrip("\\").rstrip())
        if not m:
            continue
        cmd = m.group(1).strip()
        cmds.append({"file": rel, "line": no, "command": cmd})
        for fm in FLAG.finditer(cmd):
            name, val = fm.group(1), fm.group(2)
            if val is None:
                continue
            v = mini.scalar(val.strip("'\""))
            if canon_of(name) or isinstance(v, (int, float)):
                facts.append(fact(name, v, rel, no, kind, cmd[:160], callee=re.sub(r"\S+\.(?:yaml|yml|json|toml)", "", cmd)[:120]))
    return facts, cmds


LOG_KV = re.compile(r"(?<![\w.])([A-Za-z][\w.\- ]{0,30}?)\s*[:=]\s*([-+]?\d[\d.,]*(?:[eE][-+]?\d+)?%?|true|false|True|False|[A-Za-z][\w./\-+]*)(?=[\s,;)\]]|$)")
METRIC = re.compile(r"\b(test|val(?:id(?:ation)?)?|eval(?:uation)?|train(?:ing)?|best|final|dev)[ _\-]*(acc(?:uracy)?|loss|f1(?:[ _-]?score)?|auc|map|mae|rmse|mse|bleu|rouge[-\w]*|precision|recall|iou|miou|dice|r2|perplexity|ppl|top-?[15](?:[ _-]?acc)?|error|cer|wer|psnr|ssim|nse|kge)\b\s*(?:\([^)]*\))?\s*[:=]?\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(%?)", re.I)
EPOCH = re.compile(r"\bepoch\s*[:\[]?\s*(\d+)\s*(?:/|of)\s*(\d+)", re.I)
EPOCH1 = re.compile(r"\bepoch\s*[:\[]?\s*(\d+)\b", re.I)
GPU = re.compile(r"\b((?:NVIDIA\s+)?(?:GeForce\s+)?(?:Tesla\s+|Quadro\s+)?(?:A100|A800|A40|A30|A10G?|H100|H800|V100|P100|T4|L40S?|RTX\s*A?\d{3,4}\w*|RTX\s*\d{4}\s*(?:Ti|SUPER)?|GTX\s*\d{3,4}\s*(?:Ti)?|TITAN\s*\w+|4090|3090|2080\s*Ti)(?:-?SXM\d?)?(?:[- ]\d+\s*GB)?(?:\s*PCIe)?)", re.I)
PKGVER = re.compile(r"(?<![\w.-])(%s)(?:\s*(?:version|ver\.?|v|版本)\s*(?:为|是)?\s*)?\s*(?:==|>=|~=|=|:|\s)\s*v?(\d+(?:\.\d+){1,3}(?:\+\w+)?)" % PKGS, re.I)
JAXVER = re.compile(r"\b(%s)\s+(\d+(?:\.\d+){1,3})" % PKGS, re.I)


GPUCOUNT = r"(?<![:\d.])\b(\d+)\s*(?:x\s*|块|张|个|台)?\s*(?:(?:NVIDIA|Tesla|GeForce) )?(?:(?=[A-Za-z0-9\-]*\d)[A-Za-z0-9\-]+\s*)?(?:GPUs?|gpus|显卡)"


def from_log(text, rel):
    facts, results, cmds = [], [], []
    epochs, max_epoch, total_epoch = set(), 0, 0
    series = defaultdict(list)
    for no, ln in enumerate(text.splitlines(), 1):
        s = ln.strip()
        if not s:
            continue
        m = CMD.match(s)
        if m:
            f2, c2 = from_commands(s, rel, "cli")
            for f in f2:
                f["line"] = no
            facts += f2
            for c in c2:
                c["line"] = no
            cmds += c2
        me = EPOCH.search(s)
        if me:
            max_epoch = max(max_epoch, int(me.group(1)))
            total_epoch = max(total_epoch, int(me.group(2)))
        else:
            m1 = EPOCH1.search(s)
            if m1:
                max_epoch = max(max_epoch, int(m1.group(1)))
        if no <= 600:
            facts += stage_sizes(s, rel, no, "log")
        for mm in re.finditer(r"Namespace\((.*?)\)\s*$", s):
            for k, v in re.findall(r"(\w+)=('[^']*'|\"[^\"]*\"|\[[^\]]*\]|[^,]+)", mm.group(1)):
                val = mini.flow(v.strip()) if v.strip()[:1] == "[" else mini.scalar(v.strip())
                if val is not None:
                    facts.append(fact(k, val, rel, no, "log", s[:160]))
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                for p, v in mini.flatten(obj):
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        series[p].append((no, v))
                    elif isinstance(v, str) and canon_of(p):
                        facts.append(fact(p, v, rel, no, "log", s[:120]))
                continue
            except ValueError:
                pass
        for mm in METRIC.finditer(s):
            stage, metric, val, pct = mm.group(1).lower(), mm.group(2).lower(), mm.group(3), mm.group(4)
            stage = {"valid": "val", "validation": "val", "evaluation": "test", "eval": "test", "training": "train", "dev": "val"}.get(stage, stage)
            results.append({"name": "%s_%s" % (stage, re.sub(r"[ _-]+", "_", metric)), "value": float(val), "line": no, "pct": bool(pct), "ctx": s[:160]})
        for mm in LOG_KV.finditer(s):
            k, v = mm.group(1).strip(), mm.group(2)
            kn = re.sub(r"\s+", "_", k.lower())
            if canon_of(kn) == "loss":
                try:
                    results.append({"name": "train_loss", "value": float(v.rstrip(",")), "line": no, "pct": False, "ctx": s[:160]})
                except ValueError:
                    pass
            elif canon_of(kn):
                val = mini.scalar(v.rstrip(","))
                if isinstance(val, str) and re.fullmatch(r"[\d.,]+", val):
                    val = to_num(val)
                if val is not None:
                    if isinstance(val, (int, float)) and canon_of(kn) in ("learning_rate",):
                        series[kn].append((no, val))
                    else:
                        facts.append(fact(kn, val, rel, no, "log", s[:160]))
        for mm in PKGVER.finditer(s):
            facts.append(fact("env." + mm.group(1).lower(), mm.group(2), rel, no, "env", s[:140], scope="env"))
        for mm in GPU.finditer(s):
            facts.append(fact("hardware.gpu", re.sub(r"\s+", " ", mm.group(1)).strip(), rel, no, "env", s[:140], scope="env"))
        mc = re.search(GPUCOUNT, s) or re.search(r"GPU[^\n]*?\bx\s*(\d+)\s*$", s)
        if mc and int(mc.group(1)) < 1025:
            facts.append(fact("hardware.num_gpus", int(mc.group(1)), rel, no, "env", s[:140], scope="env"))
        mc = re.search(r"\b(Intel\(R\)[^,;\n]{3,60}|AMD (?:EPYC|Ryzen)[^,;\n]{3,40}|Xeon[^,;\n]{3,40})", s)
        if mc:
            facts.append(fact("hardware.cpu", mc.group(1).strip(), rel, no, "env", s[:140], scope="env"))
        mc = re.search(r"\b(?:python|Python)\s+(3\.\d+(?:\.\d+)?)", s)
        if mc:
            facts.append(fact("env.python", mc.group(1), rel, no, "env", s[:140], scope="env"))
    if max_epoch:
        facts.append(fact("epochs_logged", max_epoch, rel, None, "log", "last epoch number seen in the log", scope="train"))
    if total_epoch:
        facts.append(fact("epochs", total_epoch, rel, None, "log", "'epoch i/N' lines: N", scope="train"))
    for p, vals in series.items():
        distinct = sorted({v for _, v in vals})
        if len(distinct) == 1:
            facts.append(fact(p, distinct[0], rel, vals[0][0], "log", "constant over %d log lines" % len(vals)))
        else:
            facts.append(fact(p + "_first", vals[0][1], rel, vals[0][0], "log", "first logged value of a series (%d points)" % len(vals)))
            facts.append(fact(p + "_last", vals[-1][1], rel, vals[-1][0], "log", "last logged value of a series (%d points)" % len(vals)))
    return facts, results, cmds


def result_facts(results, rel):
    """last and best-looking value per metric; every value is kept as a fact too (the text may quote any epoch)"""
    out = []
    by = defaultdict(list)
    for r in results:
        by[r["name"]].append(r)
    for name, rs in by.items():
        for i, r in enumerate(rs):
            out.append(fact(name, r["value"], rel, r["line"], "result", r["ctx"], scope=name.split("_")[0] if name.split("_")[0] in ("test", "val", "train", "best", "final") else "general"))
        out[-1]["context"] = (out[-1]["context"] + " [last of %d]" % len(rs))[:200]
    return out


PROSE_PARAM = None


def prose_params():
    global PROSE_PARAM
    if PROSE_PARAM is None:
        num = r"(?P<v>[-+]?\d[\d,]*(?:\.\d+)?(?:\s*[×x]\s*10\s*\^?\s*\{?-?\d+\}?|[eE][-+]?\d+)?%?)"
        sep = r"(?:\s*(?:of|is|was|were|=|:|to|at|set to|set at|fixed at|fixed to|initialised at|initialized at|equal to|为|是|设为|设置为|取|取值为|采用|使用)\s*)?"
        PROSE_PARAM = []
        for k, rx in PHRASE_RX.items():
            if PARAMS[k][2] in ("num", "any"):
                PROSE_PARAM.append((k, re.compile(r"(?:%s)%s\s*%s" % (PARAMS[k][1], sep, num), re.I)))
        PROSE_PARAM.append(("epochs", re.compile(num.replace("v", "v") + r"\s*(?:training\s+)?epochs", re.I)))
        PROSE_PARAM.append(("batch_size", re.compile(r"(?:batch(?:es)? of|mini-batches? of)\s*" + num, re.I)))
        PROSE_PARAM.append(("seed", re.compile(r"seeds?\s*\{?\s*(?P<v>\d+(?:\s*,\s*\d+)*)", re.I)))
    return PROSE_PARAM


SIZE = re.compile(r"(?P<v>\d[\d,]*(?:\.\d+)?)\s*(?P<u>[kKmM]?)\s*(?:[A-Za-z\-]+ )?(?P<n>images?|samples?|sentences?|examples?|patients?|tiles?|patches|regions?|areas|scenes?|documents?|tokens|frames?|trajectories|molecules|graphs?|nodes|cases|subjects|participants|records|rows|sequences|pairs|classes|categories|stations|sites|wells|spectra|volumes?|slices|videos?|clips|utterances|speakers|hours|圆|幅|张|个样本|条|例)", re.I)
SPLIT = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*[:/：]\s*(\d{1,3}(?:\.\d+)?)\s*[:/：]\s*(\d{1,3}(?:\.\d+)?)(?![\d.])")
SPLIT2 = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%\s*[/,]\s*(\d{1,3}(?:\.\d+)?)\s*%\s*[/,]\s*(\d{1,3}(?:\.\d+)?)\s*%")


def clause_scope(sent, pos):
    """the stage a parameter mention belongs to, from the clause it sits in (split at , ; 。 and ，；)"""
    a = max([sent.rfind(c, 0, pos) for c in ",;，；。"] + [-1]) + 1
    b = min([x for x in (sent.find(c, pos) for c in ",;，；。") if x >= 0] + [len(sent)])
    sc = scope_of(sent[a:b])
    return sc if sc in ("train", "val", "test") else "general"


METRIC_WORDS = r"F1(?:[ _-]?score)?|accuracy|acc|precision|recall|AUC|AUROC|IoU|mIoU|Dice|RMSE|MAE|MSE|BLEU|ROUGE-?\w*|perplexity|R2|NSE|准确率|精度|召回率|精确率|F1值|F1分数|均方根误差|平均绝对误差|交并比"
METRIC_PROSE = re.compile(r"(?<![A-Za-z])(?P<m>%s)\s*(?:值|分数)?\s*(?:为|是|达到|达|=|:|of|is|was|reached|reaches|achieves?|achieved|at|to)?\s*(?P<v>\d+(?:\.\d+)?)(?P<p>\s?%%)?" % METRIC_WORDS, re.I)

DATASET_NAME = re.compile(r"(?:on|using|use|from|evaluate[sd]? on)\s+(?:the\s+)?([A-Z][\w\-.]*(?:\s[A-Z][\w\-.]*)?)\s+(?:dataset|benchmark|corpus|data set)|(?:在|使用|采用)\s*([A-Za-z][\w\-]*(?:\s[A-Za-z][\w\-]*)?)\s*(?:数据集|数据)")

STAGE_SIZE = re.compile(r"\b(train(?:ing)?|val(?:id(?:ation)?)?|test(?:ing)?|dev)(?: set| split| data)?\s*[:=(]?\s*(?P<v>\d[\d,]*)\s*(?:[A-Za-z\-]+ )?(?:samples?|images?|examples?|sentences?|tiles?|patches|instances|records|rows|items|sequences|pairs)", re.I)


def stage_sizes(sent, rel, no, kind):
    out = []
    for m in STAGE_SIZE.finditer(sent):
        st = {"training": "train", "validation": "val", "valid": "val", "testing": "test", "dev": "val"}.get(m.group(1).lower(), m.group(1).lower())
        out.append(fact("dataset.size." + st, to_num(m.group("v")), rel, no, kind, sent[:160], scope="data"))
    return out


def from_prose(text, rel, kind="doc"):
    facts = []
    for no, ln in enumerate(text.splitlines(), 1):
        if not ln.strip():
            continue
        for sent in nlib.split_sentences(ln) or [ln]:
            for k, rx in prose_params():
                for m in rx.finditer(sent):
                    v = m.group("v").strip()
                    if k == "seed":
                        for x in re.findall(r"\d+", v):
                            facts.append(fact("seed", int(x), rel, no, kind, sent[:160]))
                        continue
                    n = to_num(v)
                    if n is None:
                        continue
                    facts.append(fact(k, n if not v.endswith("%") else v, rel, no, kind, sent[:160], scope=clause_scope(sent, m.start())))
            for m in OPT_RX.finditer(sent):
                if re.search(r"optimi[sz]er|优化器|trained with|using|used|采用|使用", sent, re.I):
                    facts.append(fact("optimizer", m.group(1), rel, no, kind, sent[:160]))
            facts += stage_sizes(sent, rel, no, kind)
            for m in DATASET_NAME.finditer(sent):
                nm = (m.group(1) or m.group(2) or "").strip()
                if nm and nm.lower() not in ("the", "this", "our", "a"):
                    facts.append(fact("dataset.name", nm, rel, no, kind, sent[:160], scope="data"))
            for m in METRIC_PROSE.finditer(sent):
                v = float(m.group("v"))
                facts.append(fact(re.sub(r"[ _-]+", "_", m.group("m").lower()), v / 100.0 if m.group("p") else v, rel, no, "result", sent[:160], scope=clause_scope(sent, m.start())))
            for m in SIZE.finditer(sent):
                v = to_num(m.group("v"))
                if v is None:
                    continue
                mult = {"k": 1e3, "m": 1e6}.get(m.group("u").lower(), 1)
                noun = re.sub(r"s$", "", m.group("n").lower())
                facts.append(fact("dataset.size." + ("item" if re.search(r"[一-鿿]", noun) else noun), v * mult, rel, no, kind, sent[:160], scope="data"))
            for rx in (SPLIT, SPLIT2):
                for m in rx.finditer(sent):
                    vals = [float(x) for x in m.groups()]
                    if abs(sum(vals) - 100) < 1.01 or abs(sum(vals) - 10) < 0.101 or abs(sum(vals) - 1) < 0.0101:
                        facts.append(fact("dataset.split_ratio", ":".join(m.groups()), rel, no, kind, sent[:160], scope="data"))
            for m in PKGVER.finditer(sent):
                facts.append(fact("env." + m.group(1).lower(), m.group(2), rel, no, "env", sent[:140], scope="env"))
            for m in GPU.finditer(sent):
                facts.append(fact("hardware.gpu", re.sub(r"\s+", " ", m.group(1)).strip(), rel, no, "env", sent[:140], scope="env"))
            mc = re.search(GPUCOUNT, sent)
            if mc and int(mc.group(1)) < 1025:
                facts.append(fact("hardware.num_gpus", int(mc.group(1)), rel, no, "env", sent[:140], scope="env"))
    return dedup(facts)


# ------------------------------------------------------------------------------------------------ environment files, tables
def from_requirements(text, rel):
    facts = []
    for no, ln in enumerate(text.splitlines(), 1):
        s = ln.split("#")[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*(==|>=|<=|~=|=)\s*([0-9][\w.+\-]*)", s)
        if m:
            facts.append(fact("env." + m.group(1).lower(), m.group(3) if m.group(2) in ("==", "=") else m.group(2) + m.group(3), rel, no, "env", s, scope="env"))
        m = re.match(r"^\s*-\s*([A-Za-z0-9_.\-]+)\s*(?:=|==)\s*([0-9][\w.+\-]*)", ln)
        if m:
            facts.append(fact("env." + m.group(1).lower(), m.group(2), rel, no, "env", s, scope="env"))
        m = re.match(r"^\s*(?:FROM)\s+(\S+)", ln)
        if m:
            facts.append(fact("env.docker_image", m.group(1), rel, no, "env", s, scope="env"))
            for pk, ver in re.findall(r"(pytorch|cuda|tensorflow|python)[-:]?(\d+(?:\.\d+)+)", m.group(1), re.I):
                facts.append(fact("env." + pk.lower(), ver, rel, no, "env", s, scope="env"))
        m = re.match(r"^\s*python\s*[:=]\s*[\"']?(\d+\.\d+(?:\.\d+)?)", ln, re.I)
        if m:
            facts.append(fact("env.python", m.group(1), rel, no, "env", s, scope="env"))
    return facts


def from_csv(text, rel, limit=400):
    facts, tables = [], []
    try:
        rows = list(csv.reader(io.StringIO(text), dialect=csv.Sniffer().sniff(text[:2000], delimiters=",;\t")))
    except Exception:
        rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        return facts, tables
    head = [h.strip() for h in rows[0]]
    n = 0
    for ri, row in enumerate(rows[1:], 2):
        for h, cell in zip(head, row):
            v = to_num(cell.strip())
            if v is not None and n < limit:
                facts.append(fact(h, v, rel, ri, "result", "table %s row %d" % (os.path.basename(rel), ri - 1), scope="general"))
                n += 1
    tables.append({"file": rel, "columns": head, "rows": len(rows) - 1})
    facts.append(fact("table_rows." + os.path.basename(rel), len(rows) - 1, rel, None, "data", "number of data rows in %s" % os.path.basename(rel), scope="data"))
    return facts, tables


# ------------------------------------------------------------------------------------------------ dispatch
CONFIG_EXT = (".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf", ".env")
CODE_EXT = (".py",)
DOC_EXT = (".md", ".markdown", ".txt", ".rst", ".docx", ".pdf", ".tex")
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "output", "work", ".work", ".ipynb_checkpoints", "site-packages", ".idea", ".vscode"}
LOGLIKE = re.compile(r"(?im)^.*(epoch|iter(?:ation)?|step)\s*[:\[]?\s*\d+.*(loss|acc|lr)|namespace\(|^\s*\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}|\bloss\b\s*[:=]\s*\d")


def classify(path, text):
    low = path.lower()
    base = os.path.basename(low)
    if re.match(r"requirements.*\.txt$|constraints.*\.txt$|environment\.ya?ml$|pipfile$|dockerfile.*$|conda.*\.ya?ml$|pyproject\.toml$", base):
        return "env"
    if low.endswith(".ipynb"):
        return "notebook"
    if low.endswith(".py"):
        return "code"
    if low.endswith((".sh", ".bash", ".slurm", ".sbatch")) or base in ("makefile", "run", "train.sh"):
        return "script"
    if low.endswith(".csv") or low.endswith(".tsv"):
        return "table"
    if low.endswith(".jsonl"):
        return "log"
    if low.endswith(".json") or low.endswith((".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env")) or base.startswith(".env"):
        return "config"
    if low.endswith((".log", ".out", ".err")) or LOGLIKE.search(text[:20000]):
        return "log"
    if low.endswith(DOC_EXT):
        return "doc"
    return "other"


def read_text(path):
    low = path.lower()
    if low.endswith((".docx", ".pdf")):
        import segment
        import tempfile
        d = tempfile.mkdtemp()
        doc = segment.build(path, d)
        return "\n\n".join(p["text"] for p in doc["paragraphs"] if p["text"].strip()), doc
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read(), None


def extract_file(path, rel):
    """-> {"kind", "facts", "commands", "tables", "note", "text"}"""
    try:
        text, doc = read_text(path)
    except SystemExit as e:
        return {"kind": "doc", "facts": [], "commands": [], "tables": [], "note": "could not be read: %s" % str(e)[:100], "text": ""}
    kind = classify(path, text)
    res = {"kind": kind, "facts": [], "commands": [], "tables": [], "note": "", "text": text if kind == "doc" else ""}
    if kind == "config":
        res["facts"], res["note"] = from_config(path, text, rel)
        if rel.lower().endswith(".json") and not res["facts"] and text.strip():
            pass
    elif kind == "code":
        res["facts"], res["note"] = from_python(text, rel)
    elif kind == "notebook":
        facts, outs, note = from_notebook(text, rel)
        res["facts"], res["note"] = facts, note
        if outs:
            lf, results, cmds = from_log(outs, rel)
            res["facts"] += lf + result_facts(results, rel)
    elif kind == "script":
        res["facts"], res["commands"] = from_commands(text, rel)
        res["facts"] += from_prose("\n".join(l for l in text.splitlines() if l.strip().startswith("#")), rel, "doc")
    elif kind == "log":
        lf, results, cmds = from_log(text, rel)
        res["facts"], res["commands"] = lf + result_facts(results, rel), cmds
    elif kind == "table":
        res["facts"], res["tables"] = from_csv(text, rel)
    elif kind == "env":
        res["facts"] = from_requirements(text, rel)
        if rel.lower().endswith((".yml", ".yaml", ".toml")) and not res["facts"]:
            res["facts"], res["note"] = from_config(path, text, rel)
    elif kind == "doc":
        res["facts"] = from_prose(text, rel, "doc")
        fc, cmds = from_commands(text, rel, "doc")
        res["facts"] += fc
        res["commands"] = cmds
    res["facts"] = dedup(res["facts"])
    return res


def assign_ids(facts):
    for i, f in enumerate(facts, 1):
        f["id"] = "F%03d" % i
        f["source"] = "%s:%s" % (f["file"], f["line"]) if f["line"] else f["file"]
    return facts
