from __future__ import annotations

import re
from typing import Dict, List, Set

# Bidirectional / phrase-level vocabulary mismatch rules for fast deterministic query expansion.
_PHRASE_REWRITES: List[tuple[str, str]] = [
    (r"\bflashing\b", "blinking"),
    (r"\bblinking\b", "flashing"),
    (r"\blight\b", "LED indicator"),
    (r"\blights\b", "LED indicators"),
    (r"\bwipe\b", "factory reset erase"),
    (r"\berase\b", "factory reset wipe"),
    (r"\bshutting\s+(?:itself\s+)?off\b", "shuts down overheating"),
    (r"\bturning\s+(?:itself\s+)?off\b", "shuts down overheating"),
    (r"\bmoney\s+back\b", "refund return"),
    (r"\brefund\b", "money back return"),
    (r"\bcollect\b", "pick up"),
    (r"\bpick\s+up\b", "collect"),
    (r"\banother\s+library\b", "Interlibrary Loan ILL"),
    (r"\bhead\s+protector\b", "helmet"),
    (r"\bhelmet\b", "head protector"),
    (r"\bwifi\b", "Wi-Fi"),
]

# Stem-level synonym groups for cross-interaction reranking equivalence
_SYNONYM_GROUPS: List[Set[str]] = [
    {"flash", "blink"},
    {"light", "led", "indic"},
    {"wip", "eras", "reset"},
    {"collect", "pick", "pickup"},
    {"refund", "return", "money", "back"},
    {"helmet", "protector"},
    {"overheat", "shut", "shutt", "shutdown"},
]

_TOKEN_EQUIV: Dict[str, Set[str]] = {}
for _grp in _SYNONYM_GROUPS:
    for _tok in _grp:
        _TOKEN_EQUIV[_tok] = _grp


def get_synonym_stems(stem: str) -> Set[str]:
    """Return equivalent stems (including `stem` itself) for lexical cross-interaction scoring."""
    return _TOKEN_EQUIV.get(stem, {stem})


def expand_query(query: str, max_rewrites: int = 3) -> List[str]:
    """
    Generate up to `max_rewrites` alternative phrasings for vocabulary-mismatched queries.
    Returns only new rewrites (not including the original `query`).
    """
    q = (query or "").strip()
    if not q:
        return []

    rewrites: List[str] = []
    seen = {q.lower()}

    # 1. Combined rewrite applying all matching phrase substitutions
    combined = q
    for pattern, repl in _PHRASE_REWRITES:
        combined = re.sub(pattern, repl, combined, flags=re.IGNORECASE)
    if combined.lower() not in seen:
        seen.add(combined.lower())
        rewrites.append(combined)

    # 2. Focused single-concept rewrites for RRF diversity
    for pattern, repl in _PHRASE_REWRITES:
        if len(rewrites) >= max_rewrites:
            break
        if re.search(pattern, q, flags=re.IGNORECASE):
            single = re.sub(pattern, repl, q, flags=re.IGNORECASE)
            if single.lower() not in seen:
                seen.add(single.lower())
                rewrites.append(single)

    return rewrites[:max_rewrites]
