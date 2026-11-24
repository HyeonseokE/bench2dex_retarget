"""Bundle the per-task viewers into one site: results/viewer/site/ with a hub page listing every task.

  results/viewer/site/index.html            hub: one row per task (source -> target, verdict, time, note)
  results/viewer/site/<scene>/index.html    that task's viewer (doctype and a back link added)
  results/viewer/site/<scene>/data/...      its data

  python tools/retarget/viewer/build_hub.py [--notes notes.json]

notes.json (optional): {"<task id>": "one-line note shown under the verdict"}.
"""

import argparse
import html
import json
import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VIEW = ROOT / "results" / "viewer"
RUNS = ROOT / "results"
HAND = {"rh56dfx": "Inspire RH56DFX", "rh5dg2": "Inspire RH5DG2", "shadow": "Shadow Hand",
        "schunk": "Schunk SVH", "wuji": "Wuji"}
ORDER = ["06", "12", "42", "07", "34", "60", "43", "76", "08", "44", "21", "27"]
OK = {"recorded", "rendered", "uploaded", "render_failed"}
HEAD = ('<!doctype html>\n<html lang="ko">\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        # sub-pages are served without the artifact skeleton, whose reset hides [hidden]; without it the
        # viewer's `.loading { display: grid }` keeps the loading overlay on top of the 3D view
        '<style>[hidden] { display: none !important; }</style>\n')
BACK = ('<a href="../index.html" style="font-size:13px;color:var(--ink-2);text-decoration:none">'
        '&larr; 전체 task</a>\n      ')


def target_result(scene: str, tgt: str):
    d = RUNS / scene / "ep000" / tgt
    st = json.load(open(d / "status.json")) if (d / "status.json").exists() else {}
    runs = [json.load(open(p)) for p in sorted(d.glob("spider_a*.json")) if "_record" not in p.name]
    state = st.get("state", "running")
    if state in OK:
        verdict, cls = "성공", "ok"
    elif state in ("failed_all_attempts", "stage2_failed"):
        verdict, cls = "실패", "no"
    else:
        verdict, cls = "진행 중", "run"
    return {"name": tgt, "verdict": verdict, "cls": cls,
            "minutes": sum(r.get("minutes", 0) for r in runs) or None,
            "timeout": any("rc=124" in str(a.get("error", "")) for a in st.get("attempts", [])),
            "has_run": bool(runs)}


def task_row(scene_dir: Path, notes: dict):
    s = json.load(open(scene_dir / "data" / "summary.json"))
    scene = scene_dir.name
    tid = scene[:2]
    tgts = [target_result(scene, t["name"]) for t in s["targets"]]
    cls = "ok" if any(t["cls"] == "ok" for t in tgts) else "run" if any(t["cls"] == "run" for t in tgts) else "no"
    desc = (yaml.safe_load(open(ROOT / "scenes" / f"{scene}.yaml")) or {}).get("description", "").strip()
    return {"tid": tid, "scene": scene, "name": scene[3:].replace("_", " "), "desc": desc,
            "src": s["source"]["name"], "targets": tgts, "cls": cls,
            "objects": [o["label"] for o in s["objects"]],
            "notes": [(t["name"], notes[f"{tid}:{t['name']}"]) for t in tgts if f"{tid}:{t['name']}" in notes]
                     + ([("", notes[tid])] if tid in notes else [])}


def hub_html(rows):
    n_ok = sum(r["cls"] == "ok" for r in rows)   # a task counts as solved when any target hand succeeded
    n_no = sum(r["cls"] == "no" for r in rows)
    n_run = len(rows) - n_ok - n_no
    items = []
    for r in rows:
        res = []
        for t in r["targets"]:
            meta = []
            if t["minutes"]:
                meta.append(f'{t["minutes"]:.0f}분')
            if t["timeout"]:
                meta.append("6시간 초과")
            meta.append("물리 rollout" if t["has_run"] else "기구학 경로만")
            res.append(f'<span class="tr"><span class="th">{HAND[t["name"]]}</span><span class="pill {t["cls"]}">{t["verdict"]}</span>'
                       f'<span class="meta">{" · ".join(meta)}</span></span>')
        notes = "".join(f'<span class="note">{(HAND[n] + ": ") if n else ""}{html.escape(v)}</span>' for n, v in r["notes"])
        items.append(f'''
    <a class="task" href="{r["scene"]}/index.html">
      <span class="tid">{r["tid"]}</span>
      <span class="body">
        <span class="nm">{html.escape(r["name"])}</span>
        <span class="hands">소스 {HAND[r["src"]]} <span class="arr">→</span> {" · ".join(HAND[t["name"]] for t in r["targets"])}</span>
        <span class="desc">{html.escape(r["desc"])}</span>
        <span class="objs">{" · ".join(html.escape(o) for o in r["objects"])}</span>
        {notes}
      </span>
      <span class="res">{"".join(res)}</span>
    </a>''')
    return f'''<title>Cross-Hand Retarget Check</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Hub: one row per task, same instrument-panel tokens as the per-task viewers. */
:root {{
  --bg: #f3f4f2; --panel: #fbfbfa; --line: #d9dbd6; --ink: #15171a; --ink-2: #4f5357; --ink-3: #7c8086;
  --tgt: #2a78d6; --good: #0f7d4b; --bad: #c93b3a; --sel: #e3ecf8;
  --f-display: "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
  --f-body: "IBM Plex Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", system-ui, sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #121416; --panel: #1a1d20; --line: #2c3035; --ink: #eef0f2; --ink-2: #b4b9bf; --ink-3: #868c93;
  --tgt: #3987e5; --good: #3fbf7f; --bad: #e66767; --sel: #1f2c3d; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #121416; --panel: #1a1d20; --line: #2c3035; --ink: #eef0f2; --ink-2: #b4b9bf; --ink-3: #868c93;
  --tgt: #3987e5; --good: #3fbf7f; --bad: #e66767; --sel: #1f2c3d; color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--ink); font-family: var(--f-body); font-size: 14px; line-height: 1.5; }}
.wrap {{ padding-inline: 16px; padding-block: 20px 40px; max-width: 1080px; margin: 0 auto; display: grid; gap: 16px; }}
h1 {{ font-family: var(--f-display); font-weight: 600; font-size: 26px; margin: 0; text-wrap: balance; }}
.sub {{ color: var(--ink-2); margin: 4px 0 0; max-width: 70ch; }}
.counts {{ display: flex; flex-wrap: wrap; gap: 8px 20px; font-family: var(--f-mono); font-size: 13px; color: var(--ink-2); }}
.counts b {{ font-size: 20px; font-variant-numeric: tabular-nums; margin-right: 4px; }}
.counts .ok b {{ color: var(--good); }} .counts .no b {{ color: var(--bad); }} .counts .run b {{ color: var(--tgt); }}
.list {{ display: grid; gap: 8px; }}
.task {{ display: grid; grid-template-columns: 44px minmax(0, 1fr) auto; gap: 4px 14px; align-items: start;
        padding: 12px 14px; background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
        color: inherit; text-decoration: none; }}
.task:hover {{ border-color: var(--ink-3); }}
.task:focus-visible {{ outline: 2px solid var(--tgt); outline-offset: 1px; }}
.tid {{ font-family: var(--f-display); font-size: 22px; font-weight: 600; color: var(--ink-3); line-height: 1.1; }}
.body {{ display: grid; gap: 2px; min-width: 0; }}
.nm {{ font-family: var(--f-display); font-size: 17px; font-weight: 600; }}
.hands {{ font-size: 13px; color: var(--ink-2); }} .arr {{ color: var(--tgt); }}
.desc {{ font-size: 13px; color: var(--ink-2); max-width: 75ch; }}
.objs {{ font-family: var(--f-mono); font-size: 12px; color: var(--ink-3); }}
.note {{ font-size: 13px; color: var(--ink); }}
.res {{ display: grid; justify-items: end; gap: 6px; text-align: right; }}
.tr {{ display: grid; grid-template-columns: auto auto; gap: 2px 8px; justify-items: end; align-items: center; }}
.th {{ font-size: 12px; color: var(--ink-2); }}
.tr .meta {{ grid-column: 1 / -1; }}
.pill {{ font-family: var(--f-display); font-size: 13px; letter-spacing: .04em; padding: 1px 10px; border-radius: 10px; border: 1px solid; }}
.pill.ok {{ color: var(--good); border-color: var(--good); }} .pill.no {{ color: var(--bad); border-color: var(--bad); }}
.pill.run {{ color: var(--tgt); border-color: var(--tgt); }}
.meta {{ font-family: var(--f-mono); font-size: 12px; color: var(--ink-3); }}
@media (max-width: 640px) {{ .task {{ grid-template-columns: 36px minmax(0, 1fr); }} .res {{ grid-column: 2; justify-items: start; text-align: left; }} }}
</style>

<div class="wrap">
  <header>
    <h1>Cross-Hand Retarget Check</h1>
    <p class="sub">Bench2Dex의 UR5 데모 12개 task를 다른 손으로 옮긴 결과입니다. task마다 에피소드 0을 목표 손별로 SPIDER로 한 번 시도했고(먼저 Shadow, 실패한 task는 Schunk로 다시), 판정은 Bench2Dex 공식 기준(모든 조건 0.5초 유지)입니다. 행을 누르면 그 task의 3D 재생 화면이 열립니다.</p>
  </header>
  <div class="counts"><span class="ok"><b>{n_ok}</b>성공 (손 하나 이상)</span><span class="no"><b>{n_no}</b>실패</span><span class="run"><b>{n_run}</b>진행 중</span><span><b>{len(rows)}</b>task</span></div>
  <nav class="list" aria-label="task 목록">{"".join(items)}
  </nav>
</div>
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default=None)
    a = ap.parse_args()
    notes = json.load(open(a.notes)) if a.notes else {}
    site = VIEW / "site"
    if site.exists():
        shutil.rmtree(site)
    site.mkdir(parents=True)
    rows = []
    for tid in ORDER:
        hits = [d for d in VIEW.glob(f"{tid}_*") if (d / "data" / "summary.json").exists()]
        if not hits:
            continue
        d = hits[0]
        shutil.copytree(d / "data", site / d.name / "data")
        page = (d / "index.html").read_text()
        page = page.replace('<h1 id="h1">', BACK + '<h1 id="h1">', 1)
        (site / d.name / "index.html").write_text(HEAD + page)
        rows.append(task_row(d, notes))
    (site / "index.html").write_text(hub_html(rows))
    print(f"RESULT hub: {len(rows)} tasks -> {site}")


if __name__ == "__main__":
    main()
