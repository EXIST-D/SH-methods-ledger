#!/usr/bin/env python3
"""mini.py - readers for the configuration formats experiments use, standard library only (no PyYAML / tomllib needed).

parse_yaml: block mappings and sequences, flow [..] / {..}, quoted and plain scalars, comments, block scalars (| >), anchors / aliases ignored, first document only.
parse_toml: [table] / [[array]] headers, dotted keys, strings, numbers, booleans, arrays (also multi-line), inline tables.
parse_ini / parse_env: configparser sections, KEY=value lines.
Scalars keep their written form in `Num`-aware conversion: `1e-3` is a number here although strict YAML 1.1 reads it as a string (ML configs use it as a number).
flatten(obj) -> [(dotted.path, value, line)] when line numbers are known (YAML / TOML / JSON via `lines`).
"""
from __future__ import annotations

import configparser
import json
import re

NUM = re.compile(r"^[-+]?(?:\d[\d_]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")


def scalar(tok):
    t = tok.strip()
    if t == "":
        return None
    if (t[0] == t[-1] == '"' or t[0] == t[-1] == "'") and len(t) >= 2:
        q = t[0]
        body = t[1:-1]
        if q == '"':
            body = body.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
        else:
            body = body.replace("''", "'")
        return body
    low = t.lower()
    if low in ("true", "yes", "on") and low != "on":
        return True
    if low in ("false", "no", "off") and low != "off":
        return False
    if low in ("null", "~"):
        return None
    if NUM.match(t):
        s = t.replace("_", "")
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return t
    if re.match(r"^0x[0-9a-fA-F]+$", t):
        return int(t, 16)
    if t in (".inf", "+.inf"):
        return float("inf")
    return t


def strip_comment(s):
    out, q = [], None
    for i, ch in enumerate(s):
        if q:
            out.append(ch)
            if ch == q and s[i - 1] != "\\":
                q = None
        elif ch in "\"'" and (i == 0 or s[i - 1] in " \t[{,:"):
            q = ch
            out.append(ch)
        elif ch == "#" and (i == 0 or s[i - 1] in " \t"):
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def split_top(s, sep=","):
    parts, depth, q, cur = [], 0, None, []
    for i, ch in enumerate(s):
        if q:
            cur.append(ch)
            if ch == q and s[i - 1] != "\\":
                q = None
            continue
        if ch in "\"'":
            q = ch
            cur.append(ch)
        elif ch in "[{(":
            depth += 1
            cur.append(ch)
        elif ch in "]})":
            depth -= 1
            cur.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def flow(s):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        return [flow(p) for p in split_top(s[1:-1])]
    if s.startswith("{") and s.endswith("}"):
        d = {}
        for p in split_top(s[1:-1]):
            k, _, v = p.partition(":")
            d[str(scalar(k))] = flow(v)
        return d
    return scalar(s)


class Lines:
    def __init__(self, text):
        self.rows = []
        for no, raw in enumerate(text.splitlines(), 1):
            raw = raw.replace("\t", "  ")
            if raw.strip().startswith("%") and no == 1:
                continue
            self.rows.append((no, raw))
        self.i = 0

    def peek(self):
        while self.i < len(self.rows):
            no, raw = self.rows[self.i]
            body = strip_comment(raw)
            if body.strip() == "" or body.strip() in ("---", "..."):
                if body.strip() == "---" and self.i > 0 and any(strip_comment(r).strip() for _, r in self.rows[:self.i]):
                    self.i = len(self.rows)  # a second document: ignored
                    return None
                self.i += 1
                continue
            return no, body, len(body) - len(body.lstrip())
        return None


def parse_yaml(text, lines=None):
    """-> object; lines (dict path -> line number) is filled when given"""
    L = Lines(text)
    path_lines = lines if lines is not None else {}
    obj = block(L, -1, "", path_lines)
    return obj


def block(L, parent_indent, path, pl, same_ok=False):
    item = L.peek()
    if item is None:
        return None
    no, body, ind = item
    if body.lstrip()[:1] in ("[", "{") and ind > parent_indent:
        L.i += 1
        buf = body.strip()
        while (buf.count("[") + buf.count("{")) > (buf.count("]") + buf.count("}")) and L.i < len(L.rows):
            buf += " " + strip_comment(L.rows[L.i][1]).strip()
            L.i += 1
        return flow(buf)
    is_seq = body.lstrip().startswith("- ") or body.strip() == "-"
    if ind < parent_indent or (ind == parent_indent and not (same_ok and is_seq)):
        return None
    if is_seq:
        return seq(L, ind, path, pl)
    return mapping(L, ind, path, pl)


def inline_value(L, val, ind, path, no, pl, key_line_indent):
    v = val.strip()
    v = re.sub(r"^[&!][\w./-]+\s*", "", v) if v[:1] in "&!" else v
    if v == "" or v.startswith("&"):
        sub = block(L, key_line_indent, path, pl, same_ok=(not path.endswith("]")))
        return sub
    if v in ("|", ">", "|-", ">-", "|+", ">+"):
        parts = []
        while L.i < len(L.rows):
            n2, raw2 = L.rows[L.i]
            if raw2.strip() == "" or (len(raw2) - len(raw2.lstrip())) > key_line_indent:
                parts.append(raw2.strip())
                L.i += 1
            else:
                break
        return (" " if v[0] == ">" else "\n").join(p for p in parts if p != "") if parts else ""
    if v[0] in "[{":
        buf = v
        while (buf.count("[") + buf.count("{")) > (buf.count("]") + buf.count("}")) and L.i < len(L.rows):
            buf += " " + strip_comment(L.rows[L.i][1]).strip()
            L.i += 1
        return flow(buf)
    if v.startswith("*"):
        return v
    return scalar(v)


def mapping(L, ind, path, pl):
    d = {}
    while True:
        item = L.peek()
        if item is None:
            break
        no, body, i2 = item
        if i2 < ind or (i2 == ind and body.lstrip().startswith("- ")):
            break
        if i2 > ind:
            L.i += 1  # stray deeper line
            continue
        m = re.match(r"^\s*(\"[^\"]*\"|'[^']*'|[^:#\s][^:]*?|\?)\s*:(?:\s+(.*)|$)", body)
        if not m:
            L.i += 1
            continue
        key = scalar(m.group(1)) if m.group(1)[0] in "\"'" else m.group(1).strip()
        L.i += 1
        p = (path + "." if path else "") + str(key)
        pl[p] = no
        d[key] = inline_value(L, m.group(2) or "", i2, p, no, pl, i2)
    return d


def seq(L, ind, path, pl):
    out = []
    while True:
        item = L.peek()
        if item is None:
            break
        no, body, i2 = item
        if i2 != ind or not (body.lstrip().startswith("- ") or body.strip() == "-"):
            break
        rest = body.lstrip()[1:].strip()
        L.i += 1
        p = "%s[%d]" % (path, len(out))
        pl[p] = no
        if rest == "":
            out.append(block(L, ind, p, pl))
        elif re.match(r"^(\"[^\"]*\"|'[^']*'|[^\s:#\[{][^:]*?)\s*:(?:\s|$)", rest) and not rest.startswith(("[", "{")):
            # a mapping that starts on the dash line: re-read it with the dash replaced by spaces
            L.i -= 1
            no_, raw_ = L.rows[L.i]
            L.rows[L.i] = (no_, " " * (i2 + 2) + rest)
            out.append(mapping(L, i2 + 2, p, pl))
        else:
            out.append(inline_value(L, rest, i2, p, no, pl, i2))
    return out


# ------------------------------------------------------------------------------------------------ TOML
def toml_key_parts(k):
    parts, cur, q = [], [], None
    for ch in k.strip():
        if q:
            if ch == q:
                q = None
            else:
                cur.append(ch)
        elif ch in "\"'":
            q = ch
        elif ch == ".":
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return parts


def toml_value(v):
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        return [toml_value(p) for p in split_top(v[1:-1]) if p.strip()]
    if v.startswith("{") and v.endswith("}"):
        d = {}
        for p in split_top(v[1:-1]):
            k, _, val = p.partition("=")
            node = d
            parts = toml_key_parts(k)
            for x in parts[:-1]:
                node = node.setdefault(x, {})
            node[parts[-1]] = toml_value(val)
        return d
    if v[:3] in ('"""', "'''") and v[-3:] == v[:3] and len(v) >= 6:
        return v[3:-3].lstrip("\n")
    return scalar(v)


def strip_toml_comment(s):
    return strip_comment(s)


def parse_toml(text, lines=None):
    pl = lines if lines is not None else {}
    root, cur, cur_path = {}, None, ""
    cur = root
    rows = text.splitlines()
    i = 0

    def descend(parts, create_last_list=False):
        node = root
        for k, p in enumerate(parts):
            last = k == len(parts) - 1
            if last and create_last_list:
                lst = node.setdefault(p, [])
                lst.append({})
                return lst[-1]
            nxt = node.setdefault(p, {})
            if isinstance(nxt, list):
                nxt = nxt[-1] if nxt else {}
            node = nxt
        return node
    while i < len(rows):
        raw = rows[i]
        no = i + 1
        i += 1
        s = strip_toml_comment(raw).strip()
        if not s:
            continue
        m = re.match(r"^\[\[(.+)\]\]$", s)
        if m:
            cur_path = m.group(1).strip()
            cur = descend(toml_key_parts(cur_path), True)
            continue
        m = re.match(r"^\[([^\[\]]+)\]$", s)
        if m:
            cur_path = m.group(1).strip()
            cur = descend(toml_key_parts(cur_path))
            continue
        m = re.match(r"^((?:[A-Za-z0-9_\-]+|\"[^\"]*\"|'[^']*')(?:\s*\.\s*(?:[A-Za-z0-9_\-]+|\"[^\"]*\"|'[^']*'))*)\s*=\s*(.*)$", s)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if val[:3] in ('"""', "'''") and val.count(val[:3]) < 2:
            q3 = val[:3]
            buf = [val]
            while i < len(rows):
                buf.append(rows[i])
                i += 1
                if q3 in rows[i - 1]:
                    break
            val = "\n".join(buf)
        else:
            while (val.count("[") + val.count("{") > val.count("]") + val.count("}")) and i < len(rows):
                val += " " + strip_toml_comment(rows[i]).strip()
                i += 1
        node = cur
        parts = toml_key_parts(key)
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = toml_value(val)
        pl[(cur_path + "." if cur_path else "") + ".".join(parts)] = no
    return root


def parse_ini(text):
    cp = configparser.ConfigParser(interpolation=None, allow_no_value=True, strict=False)
    cp.optionxform = str
    try:
        cp.read_string(text)
    except configparser.Error:
        cp.read_string("[DEFAULT]\n" + text)
    out = {}
    for sec in cp.sections() + (["DEFAULT"] if cp.defaults() else []):
        out[sec] = {k: (scalar(v) if v is not None else None) for k, v in cp.items(sec)}
    return out


def parse_env(text):
    d = {}
    for ln in text.splitlines():
        m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", ln)
        if m:
            d[m.group(1)] = scalar(strip_comment(m.group(2)))
    return d


def flatten(obj, prefix="", out=None):
    """-> [(dotted.path, value)] for scalars and lists of scalars; lists of maps get [i] in the path"""
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, "%s.%s" % (prefix, k) if prefix else str(k), out)
    elif isinstance(obj, list):
        if obj and all(not isinstance(x, (dict, list)) for x in obj):
            out.append((prefix, obj))
        else:
            for i, v in enumerate(obj):
                flatten(v, "%s[%d]" % (prefix, i), out)
    else:
        out.append((prefix, obj))
    return out


def load_any(path, text):
    """-> (object, {path: line}) for .yaml .yml .json .toml .ini .cfg .env"""
    low = path.lower()
    pl = {}
    if low.endswith((".yaml", ".yml")):
        return parse_yaml(text, pl), pl
    if low.endswith(".json"):
        return json.loads(text), pl
    if low.endswith(".toml"):
        return parse_toml(text, pl), pl
    if low.endswith((".ini", ".cfg", ".conf")):
        return parse_ini(text), pl
    if low.endswith(".env") or low.rsplit("/", 1)[-1].startswith(".env"):
        return parse_env(text), pl
    raise ValueError("unknown config format")
