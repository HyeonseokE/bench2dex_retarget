"""Progress of every retargeting run under $B2DR_RUNS: STATUS.md (per task x target) and results.csv.

Reads <scene>/epNNN/<target>/status.json (written by run_target.py) and, when it is still running,
the last progress line of the newest stage-3 log.

  python tools/retarget/status_report.py              # once
  python tools/retarget/status_report.py --loop 300   # refresh every 5 min
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Bench2Dex root
from retarget import paths  # noqa: E402

DONE_OK = {"recorded", "rendered", "uploaded"}
DONE_BAD = {"failed_all_attempts", "stage2_failed", "render_failed", "upload_failed"}


def tail_progress(d: Path) -> str:
    logs = sorted(d.glob("run.log"))
    if not logs:
        return ""
    txt = logs[0].read_bytes()[-200_000:].decode(errors="ignore")
    hits = re.findall(r"(frame \d+/\d+|drop of \[[^\]]*\] at frame \d+[^\n]*|RESULT [^\n]*)", txt)
    return hits[-1][:110] if hits else ""


def collect(runs: Path):
    rows = []
    for st in sorted(runs.glob("*/ep[0-9][0-9][0-9]/*/status.json")):
        d = st.parent
        try:
            s = json.load(open(st))
        except (OSError, ValueError):
            continue
        att = s.get("attempts", [])
        state = s.get("state", "?")
        rows.append({"task": d.parent.parent.name, "episode": d.parent.name, "target": d.name, "state": state,
                     "attempts": len(att), "success": state in DONE_OK,
                     "done": state in DONE_OK | DONE_BAD,
                     "progress": "" if state in DONE_OK | DONE_BAD else tail_progress(d)})
    return rows


def write(runs: Path, rows):
    with open(runs / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "episode", "target", "state", "attempts", "success"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    by = defaultdict(Counter)
    for r in rows:
        by[(r["task"], r["target"])]["ok" if r["success"] else "fail" if r["done"] else "running"] += 1
    L = [f"# Retargeting status ({runs})", "", f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", "",
         f"Total: {sum(r['success'] for r in rows)} success / {sum(r['done'] and not r['success'] for r in rows)} failed / "
         f"{sum(not r['done'] for r in rows)} in progress", "",
         "| task | target | success | failed | in progress |", "|---|---|---|---|---|"]
    for (t, g), c in sorted(by.items()):
        L.append(f"| {t} | {g} | {c['ok']} | {c['fail']} | {c['running']} |")
    run = [r for r in rows if not r["done"]]
    if run:
        L += ["", "## In progress", "", "| task | episode | target | state | attempts | last event |", "|---|---|---|---|---|---|"]
        L += [f"| {r['task']} | {r['episode']} | {r['target']} | {r['state']} | {r['attempts']} | {r['progress']} |" for r in run]
    bad = [r for r in rows if r["done"] and not r["success"]]
    if bad:
        L += ["", "## Failed", "", "| task | episode | target | state | attempts |", "|---|---|---|---|---|"]
        L += [f"| {r['task']} | {r['episode']} | {r['target']} | {r['state']} | {r['attempts']} |" for r in bad]
    (runs / "STATUS.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="seconds between refreshes (0: once)")
    a = ap.parse_args()
    runs = paths.RUNS
    runs.mkdir(parents=True, exist_ok=True)
    while True:
        write(runs, collect(runs))
        if not a.loop:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
