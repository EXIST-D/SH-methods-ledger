"""evidence_words.py - content words of a sentence or a fact name (for deciding whether a common number really belongs to a fact)."""
import re

STOP = set("a an the of in on at to for from by with and or is are was were be been this that these those it its our we used use using set as per each all".split())


def words(text):
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "")
    return {w for w in re.findall(r"[a-z]{3,}|[\u4e00-\u9fff]{2}", t.lower().replace("_", " ").replace(".", " ")) if w not in STOP}
