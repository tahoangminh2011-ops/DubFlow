from __future__ import annotations
import re, unicodedata
from difflib import SequenceMatcher
from typing import Iterable, List

_SPACE = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    s = unicodedata.normalize("NFKC", str(text or "")).replace("\r", "\n")
    s = "\n".join(_SPACE.sub(" ", line).strip() for line in s.split("\n"))
    return "\n".join([x for x in s.split("\n") if x])[:500]

def compact_text(text: str) -> str:
    return re.sub(r"[^\wÀ-ỹ]", "", normalize_text(text).lower(), flags=re.UNICODE)

def similarity(a: str, b: str) -> float:
    a, b = compact_text(a), compact_text(b)
    if not a or not b: return 0.0
    return SequenceMatcher(None, a, b).ratio()

def weighted_consensus(texts: Iterable[tuple[str, float]]) -> str:
    items = [(normalize_text(t), max(0.0, float(w))) for t,w in texts if normalize_text(t)]
    if not items: return ""
    clusters: List[dict] = []
    for text, weight in sorted(items, key=lambda x: -x[1]):
        best = None; best_sim = 0.0
        for c in clusters:
            sim = similarity(text, c["anchor"])
            if sim > best_sim:
                best_sim, best = sim, c
        if best is not None and best_sim >= 0.76:
            best["items"].append((text, weight)); best["score"] += weight
            if weight >= max(w for _,w in best["items"]): best["anchor"] = text
        else:
            clusters.append({"anchor": text, "items": [(text, weight)], "score": weight})
    winner = max(clusters, key=lambda c: c["score"])
    return max(winner["items"], key=lambda x: (x[1], len(compact_text(x[0]))))[0]
