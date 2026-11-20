#!/usr/bin/env python3
"""make_docx.py - write a small .docx from a text spec (no python-docx needed). Used for the examples and tests.
spec lines: '# Heading', plain paragraph, '@n text' = auto-numbered list paragraph; `^{1,2}` inside a line becomes a superscript run."""
import re
import sys
import zipfile
from xml.sax.saxutils import escape

CT = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'
RELS = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
DRELS = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'
STY = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style></w:styles>'


def runs(text):
    out = []
    for part in re.split(r"(\^\{[^}]*\})", text):
        if not part:
            continue
        if part.startswith("^{"):
            out.append('<w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>' % escape(part[2:-1]))
        else:
            out.append('<w:r><w:t xml:space="preserve">%s</w:t></w:r>' % escape(part))
    return "".join(out)


def make(lines, dst):
    ps = []
    for ln in lines:
        if not ln.strip():
            continue
        if ln.startswith("@n "):  # an auto-numbered Word list paragraph (numPr), no label in the text
            ps.append('<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>%s</w:p>' % runs(ln[3:]))
        elif ln.startswith("# "):
            ps.append('<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>%s</w:p>' % runs(ln[2:]))
        else:
            ps.append("<w:p>%s</w:p>" % runs(ln))
    doc = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>%s</w:body></w:document>' % "".join(ps)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CT)
        z.writestr("_rels/.rels", RELS)
        z.writestr("word/document.xml", doc)
        z.writestr("word/styles.xml", STY)
        z.writestr("word/_rels/document.xml.rels", DRELS)


if __name__ == "__main__":
    make(open(sys.argv[1], encoding="utf-8").read().splitlines(), sys.argv[2])
