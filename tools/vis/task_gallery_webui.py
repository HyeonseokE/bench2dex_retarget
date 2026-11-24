#!/usr/bin/env python3
"""Bench2Dex task gallery web UI (PC side, stdlib only).

Shows every scene in ``scenes/`` with the rendered views from
``render_task_gallery.py`` (synced to the central experiments folder) plus the
Dex-team analysis in ``task_gallery_meta.json``.

    python3 tools/vis/task_gallery_webui.py --images <dir with <task>_<view>.png> [--port 8766]
"""

import argparse
import html
import json
import mimetypes
import os
import re
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCENES = REPO / "scenes"
META = Path(__file__).resolve().with_name("task_gallery_meta.json")
VIEWS = ["front", "back", "top"]
DEX_COLOR = {"Dex 필수": "#c62828", "Dex 유리": "#ef6c00", "그리퍼 가능": "#2e7d32"}


def scene_descriptions() -> dict[str, str]:
    out = {}
    for y in sorted(SCENES.glob("*.yaml")):
        desc = ""
        with open(y, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        for i, line in enumerate(lines):
            if line.startswith("description:"):
                desc = line.split(":", 1)[1].strip().strip('"')
                for nxt in lines[i + 1 :]:
                    if nxt.startswith("  ") and ":" not in nxt.split()[0]:
                        desc += " " + nxt.strip().strip('"')
                    else:
                        break
                break
        out[y.stem] = desc
    return out


def page(images: Path) -> str:
    meta = json.load(open(META, encoding="utf-8"))["tasks"] if META.exists() else {}
    descs = scene_descriptions()
    cards = []
    done = 0
    for task in sorted(descs):
        m = meta.get(task, {})
        info = {}
        jp = images / f"{task}.json"
        if jp.exists():
            try:
                info = json.load(open(jp, encoding="utf-8"))
            except Exception:
                info = {}
        imgs = []
        for v in VIEWS:
            p = images / f"{task}_{v}.png"
            if p.exists():
                imgs.append(f'<a href="/img/{task}_{v}.png" target="_blank"><img src="/img/{task}_{v}.png" loading="lazy" title="{v}"></a>')
            else:
                imgs.append(f'<div class="ph">{v}<br>렌더 대기</div>')
        if (images / f"{task}_front.png").exists():
            done += 1
        dex = m.get("dex", "")
        tags = []
        if dex:
            tags.append(f'<span class="tag" style="background:{DEX_COLOR.get(dex, "#555")}">{html.escape(dex)}</span>')
        if m.get("retarget"):
            tags.append('<span class="tag" style="background:#1565c0">★ retarget 대상</span>')
        if m.get("tool"):
            tags.append(f'<span class="tag" style="background:#6a1b9a">도구: {html.escape(m["tool"])}</span>')
        objs = ", ".join(info.get("objects", [])[:12])
        cards.append(
            f'<div class="card" data-dex="{html.escape(dex)}" data-tool="{"1" if m.get("tool") else "0"}" data-used="{"1" if str(m.get("tool","")).startswith("사용") else "0"}">'
            f'<h3>{html.escape(task)}</h3><div class="tags">{"".join(tags)}</div>'
            f'<div class="imgs">{"".join(imgs)}</div>'
            f'<p class="kr">{html.escape(m.get("kr", ""))}</p>'
            f'<p class="en">{html.escape(descs[task])}</p>'
            f'<p class="meta"><b>분류</b> {html.escape(m.get("category", ""))}<br><b>표준 Skill</b> {html.escape(m.get("skills", ""))}'
            + (f'<br><b>물체</b> {html.escape(objs)}' if objs else "")
            + "</p></div>"
        )
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Bench2Dex task gallery</title>
<meta http-equiv="refresh" content="60">
<style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}}
h1{{font-size:20px;margin:0 0 8px}} .sub{{color:#666;font-size:13px;margin-bottom:12px}}
.filters button{{margin:0 4px 8px 0;padding:4px 10px;border:1px solid #bbb;background:#fff;border-radius:4px;cursor:pointer}}
.filters button.on{{background:#333;color:#fff}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(640px,1fr));gap:14px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px}} .card.hide{{display:none}}
.card h3{{margin:0 0 6px;font-size:15px}} .tags{{margin-bottom:6px}}
.tag{{display:inline-block;color:#fff;font-size:11px;padding:2px 7px;border-radius:10px;margin:0 4px 4px 0}}
.imgs{{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}} .imgs img{{width:100%;border-radius:4px;background:#000}}
.ph{{aspect-ratio:16/9;background:#eee;color:#999;display:flex;align-items:center;justify-content:center;text-align:center;font-size:12px;border-radius:4px}}
.kr{{font-weight:600;margin:8px 0 2px}} .en{{color:#555;font-size:13px;margin:0 0 6px}} .meta{{font-size:12px;color:#444;margin:0;line-height:1.5}}
</style></head><body>
<h1>Bench2Dex task gallery — {len(descs)} scenes (공식 26 + fork 1)</h1>
<div class="sub">렌더 완료 {done}/{len(descs)} · 뷰: front(로봇 맞은편) / back(로봇 뒤) / top(위) · 이미지 클릭 = 원본 · 60초마다 자동 새로고침 · 이미지: {html.escape(str(images))}</div>
<div class="filters">
<button class="on" data-f="all">전체</button><button data-f="dex:Dex 필수">Dex 필수</button><button data-f="dex:Dex 유리">Dex 유리</button><button data-f="dex:그리퍼 가능">그리퍼 가능</button>
<button data-f="tool">도구 관련</button><button data-f="used">도구 실제 사용</button>
</div>
<div class="grid">{"".join(cards)}</div>
<script>
document.querySelectorAll('.filters button').forEach(b=>b.onclick=()=>{{
 document.querySelectorAll('.filters button').forEach(x=>x.classList.remove('on')); b.classList.add('on');
 const f=b.dataset.f; document.querySelectorAll('.card').forEach(c=>{{
  let show=true; if(f.startsWith('dex:')) show=c.dataset.dex===f.slice(4); else if(f==='tool') show=c.dataset.tool==='1'; else if(f==='used') show=c.dataset.used==='1';
  c.classList.toggle('hide',!show);}});}});
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    images = Path(a.images).resolve()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            path = urllib.parse.unquote(self.path.split("?")[0])
            if path.startswith("/img/"):
                name = os.path.basename(path[5:])
                p = images / name
                if not re.fullmatch(r"[\w.-]+\.(png|jpg|json)", name) or not p.exists():
                    self.send_error(404)
                    return
                data = p.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            body = page(images).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(f"task gallery: http://{a.host}:{a.port}/  images={images}", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
