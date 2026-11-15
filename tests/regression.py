#!/usr/bin/env python3
"""Live HTTP regression suite against BASE_URL (default http://127.0.0.1:8000)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import urllib.request
from typing import Any, Dict, List


def main() -> int:
    base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    questions_path = Path(__file__).resolve().parent / "questions.json"
    questions: List[Dict[str, Any]] = json.loads(questions_path.read_text(encoding="utf-8"))

    # Discover indexed documents so doc-scoped items can be verified accurately
    with urllib.request.urlopen(f"{base_url}/api/v1/documents", timeout=15) as r:
        indexed_docs = {d["id"] for d in json.loads(r.read().decode("utf-8"))}

    passed = 0
    total = 0
    abstain_total = 0
    abstain_passed = 0
    must_abstain_failed = False
    latencies_ms: List[float] = []

    print(f"Running regression suite against {base_url} ({len(questions)} questions)\n")
    print(f"{'Question':<64} | {'Status':<6} | {'Latency':>9} | Notes")
    print("-" * 105)

    for item in questions:
        q = item["question"]
        doc = item.get("doc")
        must_abstain = bool(item.get("must_abstain"))
        expected_contains = [str(x) for x in (item.get("expected_contains") or [])]

        if doc and doc not in indexed_docs:
            print(f"{q[:62]:<64} | {'SKIP':<6} | {'-':>9} | doc {doc} not indexed")
            continue

        total += 1
        payload: Dict[str, Any] = {"question": q}
        if doc:
            payload["document_ids"] = [doc]

        t0 = time.perf_counter()
        data: Dict[str, Any] = {}
        for attempt in range(2):
            req = urllib.request.Request(
                f"{base_url}/api/v1/ask",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if not data.get("error"):
                break
            time.sleep(3.5)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        time.sleep(0.5)
        latencies_ms.append(dt_ms)

        abstained = bool(data.get("abstained"))
        answer = str(data.get("answer") or "")
        citations = data.get("citations") or []

        ok = True
        note = ""
        if must_abstain:
            abstain_total += 1
            if not abstained or len(citations) != 0:
                ok = False
                must_abstain_failed = True
                note = f"abstained={abstained}, citations={len(citations)}"
            else:
                abstain_passed += 1
                note = "abstained (0 citations)"
        else:
            missing = [s for s in expected_contains if s.lower() not in answer.lower()]
            if abstained:
                ok = False
                note = f"false abstention ({data.get('abstain_reason')})"
            elif missing:
                ok = False
                note = f"missing {missing}"
            elif len(citations) < 1:
                ok = False
                note = "0 citations"
            else:
                note = f"{len(citations)} citation(s)"

        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"{q[:62]:<64} | {status:<6} | {dt_ms:>7.1f}ms | {note}")

    avg_lat = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    abstain_acc = (abstain_passed / abstain_total * 100.0) if abstain_total else 100.0
    print("-" * 105)
    print(
        f"Summary: {passed}/{total} passed | "
        f"Abstention accuracy: {abstain_passed}/{abstain_total} ({abstain_acc:.1f}%) | "
        f"Avg latency: {avg_lat:.1f} ms"
    )

    if must_abstain_failed:
        print("ERROR: One or more must_abstain trap questions failed!", file=sys.stderr)
        return 1
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
