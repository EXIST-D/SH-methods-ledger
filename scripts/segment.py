#!/usr/bin/env python3
"""segment.py - read a manuscript (Markdown, plain text, LaTeX, Word .docx, or a PDF's text) into paragraphs and sentences with exact positions, so that a polish
can be written back into the SAME file format with every untouched character unchanged.

Document model (JSON): {"format", "file", "paragraphs": [{"id": "P3", "kind": "text|heading|frozen", "frozen_reason", "start", "end", "text", "section", "in_refs",
                         "sentences": [{"id": "S012", "start", "end", "text"}]}]}
start / end are character offsets into the source text (md / txt / tex) or into the paragraph text (docx; the paragraph index is the paragraph id).
Frozen paragraphs (code, tables, equations, LaTeX preambles / environments, reference lists, complex Word paragraphs) are never edited; their text still counts for consistency checks.
"""
from __future__ import annotations

import os
import re
import zipfile
import xml.etree.ElementTree as ET

import nlib

REF_HEAD = re.compile(r"^\s*(?:\d+[.\s]*)?(references?|bibliography|literature cited|works cited|参考文献|致谢|acknowledge?ments?)\s*$", re.I)
TEX_FROZEN = re.compile(r"^\s*(?:\\(?:documentclass|usepackage|begin|end|section|subsection|subsubsection|chapter|paragraph|title|author|date|maketitle|bibliography|bibliographystyle|input|include|newcommand|renewcommand|def|label|caption|centering|includegraphics|tableofcontents|appendix|abstract)\b|%)")
TEX_ENV = re.compile(r"\\begin\{(equation\*?|align\*?|eqnarray\*?|gather\*?|multline\*?|figure\*?|table\*?|tabular|verbatim|lstlisting|tikzpicture|thebibliography|minted|algorithm\*?|algorithmic|displaymath)\}")


def split_para_sentences(text):
    """sentences of one paragraph with offsets inside it"""
    out, pos = [], 0
    for s in nlib.split_sentences(text):
        k = text.find(s, pos)
        if k < 0:
            k = text.find(s)
        if k < 0:
            continue
        out.append({"start": k, "end": k + len(s), "text": s})
        pos = k + len(s)
    return out


def lines_blocks(raw):
    """blank-line separated blocks of a text file -> [(start, end, text)]"""
    out = []
    for m in re.finditer(r"(?:[^\n]+(?:\n(?!\s*\n)[^\n]*)*)", raw):
        if m.group(0).strip():
            out.append((m.start(), m.end(), m.group(0)))
    return out


def load_text_like(path, ext):
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    paras, in_fence, in_refs, in_env = [], False, False, None
    sec = "(start)"
    for start, end, text in lines_blocks(raw):
        first = text.lstrip().split("\n", 1)[0]
        kind, reason, off = "text", None, 0
        body = text
        if ext == ".tex":
            m = TEX_ENV.search(text)
            if in_env or m:
                kind, reason = "frozen", "LaTeX environment"
                if m and not re.search(r"\\end\{%s\}" % re.escape(m.group(1)), text):
                    in_env = m.group(1)
                if in_env and re.search(r"\\end\{%s\}" % re.escape(in_env), text):
                    in_env = None
            elif all(TEX_FROZEN.match(l) or not l.strip() for l in text.split("\n")):
                kind, reason = "frozen", "LaTeX command / comment"
            else:
                mh = re.match(r"\s*\\(?:sub)*section\*?\{([^{}]*)\}", first)
                if mh and re.fullmatch(r"\s*\\(?:sub)*section\*?\{[^{}]*\}\s*", text):
                    sec = mh.group(1)
                    kind, reason = "frozen", "section command"
        else:
            if re.match(r"\s*(```|~~~)", first) or in_fence:
                kind, reason = "frozen", "code block"
                if len(re.findall(r"^\s*(?:```|~~~)", text, re.M)) % 2 == 1:
                    in_fence = not in_fence
            elif re.match(r"\s*---\s*$", first) and start == 0:
                kind, reason = "frozen", "front matter"
            elif re.match(r"\s*\|.*\|", first) or re.match(r"\s*\$\$", first) or re.match(r"\s*!\[", first) or re.match(r"\s*<[a-zA-Z/]", first):
                kind, reason = "frozen", "table / math / image / html"
            else:
                mh = re.match(r"(\s*#{1,6}\s+)(.+?)\s*#*\s*$", first)
                if mh and "\n" not in text.strip():
                    kind = "heading"
                    off = len(mh.group(1)) + (len(text) - len(text.lstrip()))
                    body = mh.group(2)
                    sec = body
                    if REF_HEAD.match(body):
                        in_refs = True
                    elif in_refs and re.match(r"^\s*(?:\d+[.\s]*)?[A-Z]", body):
                        in_refs = in_refs and bool(REF_HEAD.match(body))
        if in_refs and kind == "text":
            kind, reason = "frozen", "reference list / acknowledgements"
        if in_refs and kind == "frozen" and reason == "code block":
            mf = re.fullmatch(r"(\s*(?:```|~~~)[^\n]*\n)(.*?)(\n\s*(?:```|~~~)\s*)", text, re.S)
            if mf:  # a reference list fenced by a PDF-to-Markdown step: the entries are the content
                reason, off, body = "reference list (fenced)", len(mf.group(1)), mf.group(2)
        if kind == "text" and ext == ".tex":
            # command-only lines glued to the prose (\\documentclass, \\begin{document}, \\section{..}, \\end{document}, comments) stay outside the edited text
            ls = body.split("\n")
            pos, a, b = 0, 0, len(ls)
            while a < b and (TEX_FROZEN.match(ls[a]) or not ls[a].strip()):
                pos += len(ls[a]) + 1
                a += 1
            while b > a and (TEX_FROZEN.match(ls[b - 1]) or not ls[b - 1].strip()):
                b -= 1
            if a >= b:
                kind, reason = "frozen", "LaTeX command / comment"
            else:
                off += pos
                body = "\n".join(ls[a:b])
        if kind == "text":
            # list markers and block quotes stay outside the edited text
            mm = re.match(r"(\s*(?:[-*+]|\d+[.)]|>)\s+)", text)
            if mm:
                off = len(mm.group(1))
                body = text[off:]
        paras.append({"kind": kind, "frozen_reason": reason, "start": start + off, "end": start + off + len(body), "text": body, "section": sec, "in_refs": in_refs})
    return raw, paras


# ------------------------------------------------------------------------------------------------ docx
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
SIMPLE_CHILDREN = {"pPr", "r", "bookmarkStart", "bookmarkEnd", "proofErr", "rPr"}


def q(tag):
    return "{%s}%s" % (W, tag)


def docx_paragraphs(path):
    z = zipfile.ZipFile(path)
    xml = z.read("word/document.xml").decode("utf-8")
    root = ET.fromstring(xml)
    paras = []
    for p in root.iter(q("p")):
        kids = [c for c in p]
        complex_ = [c for c in kids if c.tag.split("}")[1] not in SIMPLE_CHILDREN]
        runs = [c for c in kids if c.tag == q("r")]
        text, bad = [], False
        for r in runs:
            rp = r.find(q("rPr"))
            sup = rp is not None and rp.find(q("vertAlign")) is not None and rp.find(q("vertAlign")).get(q("val")) == "superscript"
            if sup:
                text.append("<sup>")
            for e in r:
                tag = e.tag.split("}")[1]
                if tag == "t":
                    text.append(e.text or "")
                elif tag == "tab":
                    text.append("\t")
                elif tag in ("br", "cr"):
                    text.append("\n")
                elif tag in ("rPr", "lastRenderedPageBreak"):
                    continue
                elif tag in ("drawing", "pict", "object", "fldChar", "instrText", "footnoteReference", "endnoteReference", "sym", "ptab"):
                    bad = True
            if sup:
                text.append("</sup>")
        ppr = p.find(q("pPr"))
        numbered = ppr is not None and ppr.find(q("numPr")) is not None
        style = ""
        if ppr is not None and ppr.find(q("pStyle")) is not None:
            style = ppr.find(q("pStyle")).get(q("val"), "")
        paras.append({"text": "".join(text).replace("</sup><sup>", ""), "style": style, "complex": bool(complex_) or bad, "numbered": numbered})
    return paras


def load_docx(path):
    ps = docx_paragraphs(path)
    out, in_refs, sec = [], False, "(start)"
    for i, p in enumerate(ps):
        t = p["text"]
        if not t.strip():
            continue
        kind, reason = "text", None
        if re.match(r"(?i)heading|title|subtitle", p["style"]):
            kind = "heading"
            sec = t
            in_refs = bool(REF_HEAD.match(t))
        elif p["complex"]:
            kind, reason = "frozen", "Word paragraph with fields / hyperlinks / drawings / footnotes"
        if in_refs and kind == "text":
            kind, reason = "frozen", "reference list / acknowledgements"
        out.append({"kind": kind, "frozen_reason": reason, "start": 0, "end": len(t), "text": t, "section": sec, "in_refs": in_refs, "docx_index": i, "style": p["style"], "numbered": p.get("numbered", False)})
    return None, out


def load_pdf(path, work):
    import json
    import subprocess
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(work, "parsed")
    res = subprocess.run([sys.executable, os.path.join(here, "parse_pdf.py"), path, "--out", outdir], capture_output=True, text=True)
    try:
        summ = json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        raise SystemExit("PDF parsing failed: " + (res.stdout[-300:] + res.stderr[-300:]))
    if summ.get("status") == "failed":
        raise SystemExit("PDF parsing failed: %s" % json.dumps(summ, ensure_ascii=False))
    doc = json.load(open(os.path.join(outdir, "parsed.json"), encoding="utf-8"))
    parts, in_refs = [], False
    for b in doc["blocks"]:
        t = (b.get("text") or "").strip()
        if not t or b["type"] in ("header", "footer", "margin", "table", "formula"):
            continue
        if b["type"] == "heading" and REF_HEAD.match(re.sub(r"^\d+[.\s]*", "", t)):
            in_refs = True
        parts.append(("## " + t) if b["type"] in ("heading", "title") else t)
        if in_refs and b["type"] != "heading":
            parts[-1] = "```\n" + t + "\n```"
    raw = "\n\n".join(parts) + "\n"
    p = os.path.join(work, "from_pdf.md")
    open(p, "w", encoding="utf-8").write(raw)
    return p


def build(path, work):
    ext = os.path.splitext(path)[1].lower()
    fmt = {".md": "md", ".markdown": "md", ".txt": "txt", ".tex": "tex", ".docx": "docx", ".pdf": "md"}.get(ext)
    if not fmt:
        raise SystemExit("unsupported input type %s (md, txt, tex, docx, pdf)" % ext)
    src_note = None
    if ext == ".pdf":
        path = load_pdf(path, work)
        ext = ".md"
        src_note = "PDF text was extracted to Markdown (no formatting): the polish is delivered as Markdown"
    if fmt == "docx":
        raw, paras = load_docx(path)
    else:
        raw, paras = load_text_like(path, ".tex" if fmt == "tex" else ext)
    n = 0
    for i, p in enumerate(paras, 1):
        p["id"] = "P%d" % i
        p["sentences"] = []
        if p["kind"] == "frozen":
            continue
        ss = [{"start": 0, "end": len(p["text"]), "text": p["text"]}] if p["kind"] == "heading" else split_para_sentences(p["text"])
        for s in ss:
            n += 1
            s["id"] = "S%03d" % n
            p["sentences"].append(s)
    return {"format": fmt, "file": os.path.basename(path), "raw": raw, "paragraphs": paras, "note": src_note}


# ------------------------------------------------------------------------------------------------ writing back
def apply_text_like(doc, cur):
    """rebuild the source text with every edited sentence replaced in place (cur: sid -> current text); untouched characters are kept"""
    raw = doc["raw"]
    repl = []
    for p in doc["paragraphs"]:
        for s in p["sentences"]:
            new = cur.get(s["id"], s["text"])
            if new != s["text"]:
                a = p["start"] + s["start"]
                repl.append((a, a + (s["end"] - s["start"]), new))
    out, last = [], 0
    for a, b, new in sorted(repl):
        out.append(raw[last:a])
        out.append(new)
        last = b
    out.append(raw[last:])
    return "".join(out)


def paragraph_texts(doc, cur):
    """paragraph id -> (original text, new text)"""
    res = {}
    for p in doc["paragraphs"]:
        if p["kind"] == "frozen":
            continue
        t, last, out = p["text"], 0, []
        for s in p["sentences"]:
            out.append(t[last:s["start"]])
            out.append(cur.get(s["id"], s["text"]))
            last = s["end"]
        out.append(t[last:])
        res[p["id"]] = (t, "".join(out))
    return res


def word_diff(a, b):
    """[(op, text)] with op in '=', '-', '+' on word tokens (whitespace kept attached)"""
    import difflib
    tok = r"<sup>.*?</sup>|\[[\d,\s\-–]+\]|\s+|[A-Za-z0-9_'’]+|[^\sA-Za-z0-9]"
    ta, tb = re.findall(tok, a), re.findall(tok, b)
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out.append(("=", "".join(ta[i1:i2])))
        else:
            if i2 > i1:
                out.append(("-", "".join(ta[i1:i2])))
            if j2 > j1:
                out.append(("+", "".join(tb[j1:j2])))
    return out


def write_docx(src, dst, ptexts, doc, tracked, author="citation-ledger", date="2000-01-01T00:00:00Z", extras=None):
    """copy a .docx, replacing the text of edited paragraphs; tracked=True writes <w:ins>/<w:del> (Word shows the changes), False writes the accepted text.
    Only the first run's character formatting is kept inside an edited paragraph; paragraph properties are untouched."""
    z = zipfile.ZipFile(src)
    xml = z.read("word/document.xml").decode("utf-8")
    m = re.search(r"<w:document\b[^>]*>", xml)
    root_tag = m.group(0) if m else None
    for pre, uri in re.findall(r'xmlns:(\w+)="([^"]+)"', xml):
        ET.register_namespace(pre, uri)
    ET.register_namespace("", "")
    root = ET.fromstring(xml)
    changed = {}
    drop = set()
    for p in doc["paragraphs"]:
        if p["id"] in ptexts and ptexts[p["id"]][0] != ptexts[p["id"]][1]:
            changed[p["docx_index"]] = ptexts[p["id"]]
            if ptexts[p["id"]][1] is None:
                drop.add(p["docx_index"])
    rid = [1000]
    dropped_els = set()
    for i, p in enumerate(root.iter(q("p"))):
        if i in drop:
            dropped_els.add(p)
        if i not in changed:
            continue
        old, new = changed[i]
        runs = [c for c in p if c.tag == q("r")]
        rpr = None
        for r in runs:
            if r.find(q("rPr")) is not None:
                rpr = r.find(q("rPr"))
                break
        for c in list(p):
            if c.tag != q("pPr"):
                p.remove(c)

        def mk_run(text, deleted=False):
            segs = re.split(r"(<sup>.*?</sup>)", text)
            runs_ = []
            for sg in segs:
                if not sg:
                    continue
                sup = sg.startswith("<sup>") and sg.endswith("</sup>")
                r = ET.Element(q("r"))
                base = ET.fromstring(ET.tostring(rpr)) if rpr is not None else None
                if sup:
                    if base is None:
                        base = ET.Element(q("rPr"))
                    for old_ in base.findall(q("vertAlign")):
                        base.remove(old_)
                    va = ET.SubElement(base, q("vertAlign"))
                    va.set(q("val"), "superscript")
                elif base is not None:
                    for old_ in base.findall(q("vertAlign")):
                        base.remove(old_)
                if base is not None:
                    r.append(base)
                t = ET.SubElement(r, q("delText" if deleted else "t"))
                t.text = sg[5:-6] if sup else sg
                t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                runs_.append(r)
            return runs_
        if new is None:
            new = ""
        if not tracked:
            for rr in mk_run(new):
                p.append(rr)
            continue
        for op, text in word_diff(old, new):
            if op == "=":
                for rr in mk_run(text):
                    p.append(rr)
            else:
                rid[0] += 1
                wrap = ET.SubElement(p, q("del" if op == "-" else "ins"))
                wrap.set(q("id"), str(rid[0]))
                wrap.set(q("author"), author)
                wrap.set(q("date"), date)
                for rr in mk_run(text, deleted=(op == "-")):
                    wrap.append(rr)
    if drop and not tracked:
        for parent in root.iter():
            for ch in list(parent):
                if ch.tag == q("p") and ch in dropped_els:
                    parent.remove(ch)
    if extras:
        import copy
        plist = list(root.iter(q("p")))
        parents = {c: pr for pr in root.iter() for c in pr}
        for idx, texts in sorted(extras.items(), reverse=True):
            if idx >= len(plist):
                continue
            base = plist[idx]
            pr = parents.get(base)
            if pr is None:
                continue
            pos = list(pr).index(base) + 1
            rpr0 = None
            for rr in base.iter(q("r")):
                if rr.find(q("rPr")) is not None:
                    rpr0 = rr.find(q("rPr"))
                    break
            for k, txt in enumerate(texts):
                el = copy.deepcopy(base)
                for c in list(el):
                    if c.tag != q("pPr"):
                        el.remove(c)
                run = ET.SubElement(el, q("r"))
                if rpr0 is not None:
                    run.append(copy.deepcopy(rpr0))
                tt = ET.SubElement(run, q("t"))
                tt.text = txt
                tt.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                if tracked:
                    rid[0] += 1
                    ins = ET.Element(q("ins"))
                    ins.set(q("id"), str(rid[0]))
                    ins.set(q("author"), author)
                    ins.set(q("date"), date)
                    el.remove(run)
                    ins.append(run)
                    el.append(ins)
                pr.insert(pos + k, el)
    out = ET.tostring(root, encoding="unicode")
    if root_tag:
        out = re.sub(r"<w:document\b[^>]*>", lambda _m: root_tag, out, count=1)
    out = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + out
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zo:
        for item in z.infolist():
            data = z.read(item.filename)
            if item.filename == "word/document.xml":
                data = out.encode("utf-8")
            zo.writestr(item, data)
