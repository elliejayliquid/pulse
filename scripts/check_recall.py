"""Summarize passive-recall behavior from a persona's log file.

Usage:
    python scripts/check_recall.py valentine
    python scripts/check_recall.py logs/valentine.log

Parses the "Passive recall:" / "Passive recall hit:" INFO lines that
core.context emits on every recall-enabled prompt build, and prints what
future-you needs for tuning `context.recall` (min_similarity, top_k):
score distribution, hits-per-call, and the most frequently recalled
memories. No live log-watching required — the log remembers for you.
"""

import re
import sys
from collections import Counter
from pathlib import Path

STATS_RE = re.compile(
    r"Passive recall: chunks=(\d+), pool=(\d+), "
    r"hits_above_thresh=(\d+), top_final=([\d.]+), top_base=([\d.]+)"
)
HIT_RE = re.compile(
    r'Passive recall hit: score=([\d.]+) id=(\S+) "(.*)"'
)


def resolve_log(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    p = Path(__file__).resolve().parent.parent / "logs" / f"{arg}.log"
    if p.exists():
        return p
    sys.exit(f"No log found for '{arg}' (tried {p})")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    log_path = resolve_log(sys.argv[1])

    calls = []          # (hits_above_thresh, top_final)
    hit_scores = []
    hit_memories = Counter()   # id -> count
    previews = {}              # id -> latest preview text

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = STATS_RE.search(line)
        if m:
            calls.append((int(m.group(3)), float(m.group(4))))
            continue
        m = HIT_RE.search(line)
        if m:
            score, mem_id, preview = float(m.group(1)), m.group(2), m.group(3)
            hit_scores.append(score)
            hit_memories[mem_id] += 1
            previews[mem_id] = preview

    if not calls:
        sys.exit(f"No 'Passive recall:' lines in {log_path} — recall disabled, "
                 "or the log predates hit logging (2026-07-14).")

    print(f"=== Passive recall summary: {log_path.name} ===\n")
    print(f"Recall-enabled prompt builds: {len(calls)}")
    zero = sum(1 for h, _ in calls if h == 0)
    print(f"  with zero hits: {zero} ({zero * 100 // len(calls)}%)")
    avg_hits = sum(h for h, _ in calls) / len(calls)
    print(f"  avg hits above threshold: {avg_hits:.1f}")

    if hit_scores:
        print(f"\nSurfaced hits: {len(hit_scores)}")
        buckets = Counter(f"{int(s * 20) * 5 / 100:.2f}" for s in hit_scores)
        print("  score distribution (0.05 buckets):")
        for bucket in sorted(buckets):
            bar = "#" * buckets[bucket]
            print(f"    {bucket}+ {bar} ({buckets[bucket]})")
        near_floor = sum(1 for s in hit_scores if s < 0.35)
        print(f"  hits scoring < 0.35 (raise min_similarity to drop these): "
              f"{near_floor} ({near_floor * 100 // len(hit_scores)}%)")

        print("\nMost recalled memories (does he keep dredging up the same ones?):")
        for mem_id, count in hit_memories.most_common(10):
            print(f"  {count:3}x  {mem_id}  \"{previews[mem_id][:70]}\"")

    print("\nTuning knobs: context.recall.min_similarity (default 0.30), "
          "context.recall.top_k (default 6).")


if __name__ == "__main__":
    main()
