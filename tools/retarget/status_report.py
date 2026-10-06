"""Progress of every retargeting run under results/ ($B2DR_RUNS), written into experiments/:
experiments/STATUS.md (all task x target pairs) and experiments/<scene>/<target>/STATUS.md, results.csv.

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
    for d in sorted({p.parent for pat in ("status.json", "run.log") for p in runs.glob(f"*/ep[0-9][0-9][0-9]/*/{pat}")}):
        st = d / "status.json"
        try:
            s = json.load(open(st)) if st.exists() else {"state": "running"}   # first attempt still in stage 2/3
        except (OSError, ValueError):
            continue
        att = s.get("attempts", [])
        state = s.get("state", "?")
        rows.append({"task": d.parent.parent.name, "episode": d.parent.name, "target": d.name, "state": state,
                     "attempts": len(att), "success": state in DONE_OK,
                     "done": state in DONE_OK | DONE_BAD,
                     "progress": "" if state in DONE_OK | DONE_BAD else tail_progress(d)})
    return rows


def pair_table(rows):
    L = ["| episode | state | attempts | last event |", "|---|---|---|---|"]
    L += [f"| {r['episode']} | {r['state']} | {r['attempts']} | {r['progress'] or '-'} |" for r in rows]
    return L


def write(exp: Path, rows, runs: Path):
    """experiments/STATUS.md (all pairs) + experiments/<scene>/<target>/STATUS.md, results.csv."""
    stamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    fields = ["task", "episode", "target", "state", "attempts", "success"]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["target"])].append(r)
    for (task, tgt), rs in groups.items():
        d = exp / task / tgt
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rs:
                w.writerow({k: r[k] for k in fields})
        ok = sum(r["success"] for r in rs)
        bad = sum(r["done"] and not r["success"] for r in rs)
        (d / "STATUS.md").write_text("\n".join(
            [f"# {task} -> {tgt}: status", "", f"Updated: {stamp}", "",
             f"{ok} success / {bad} failed / {len(rs) - ok - bad} in progress (episodes started so far)", ""]
            + pair_table(rs)) + "\n")
    L = [f"# Retargeting status", "", f"Updated: {stamp}  ·  data: `{runs}`", "",
         f"Total: {sum(r['success'] for r in rows)} success / {sum(r['done'] and not r['success'] for r in rows)} failed / "
         f"{sum(not r['done'] for r in rows)} in progress", "",
         "| task | target | success | failed | in progress |", "|---|---|---|---|---|"]
    for (t, g), rs in sorted(groups.items()):
        c = Counter("ok" if r["success"] else "fail" if r["done"] else "running" for r in rs)
        L.append(f"| [{t}]({t}/) | [{g}]({t}/{g}/STATUS.md) | {c['ok']} | {c['fail']} | {c['running']} |")
    (exp / "STATUS.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="seconds between refreshes (0: once)")
    a = ap.parse_args()
    runs, exp = paths.RUNS, paths.EXPERIMENTS
    runs.mkdir(parents=True, exist_ok=True)
    exp.mkdir(parents=True, exist_ok=True)
    while True:
        write(exp, collect(runs), runs)
        if not a.loop:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
