"""Build the retarget web viewer's data for one task (pure Python; Isaac only via viewer_fk.py).

Layout follows the 06 -> Shadow prototype viewer (results/legacy_06_shadow/viewer): index.html next to
data/ with summary.json, meshes.json + meshes.b64.txt, and one <name>.json + <name>.b64.txt per array
bundle (float16 / int8 arrays, base64 because artifacts serve no raw binaries). Differences: several
target hands per task (a selector), physics runs are the SPIDER attempts of run_target.py, and the
condition chips are Bench2Dex's terminal sub-conditions evaluated on every rollout frame.

  python tools/retarget/viewer/build_viewer.py --task 06 --episodes 0 [--targets rh5dg2 schunk wuji]

Output: results/viewer/<scene>/ (index.html copied from this folder). FK runs once per (episode,
robot) and is cached in results/<scene>/epNNN/viewer_fk_<robot>.npz (rerun with --refresh_fk).
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from retarget import paths  # noqa: E402

HAND = {"rh56dfx": "Inspire RH56DFX", "rh5dg2": "Inspire RH5DG2", "shadow": "Shadow Hand",
        "schunk": "Schunk SVH", "wuji": "Wuji"}
STRIDE = 2              # reference animation keeps every 2nd frame (20 fps data -> 10 fps samples)
FACE_BUDGET = 800       # per link mesh after decimation


# ------------------------------------------------------------------ meshes
def decimate(v, f, budget=FACE_BUDGET):
    """Vertex clustering on a growing grid until the mesh has <= budget faces."""
    v = np.asarray(v, np.float64)
    f = np.asarray(f, np.int64)
    vv, ff = weld(v, f, 1e-6)                       # merge the duplicated corners of triangle soups
    ext = float(np.ptp(v, 0).max()) or 1e-3
    cell = ext / 400
    while len(ff) > budget and cell < ext:
        vv, ff = weld(v, f, cell)
        cell *= 1.25
    return vv, ff


def weld(v, f, cell):
    """Merge vertices on a grid of size cell; drop collapsed and duplicate faces."""
    _, inv = np.unique(np.floor(v / cell).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    ff = inv[f]
    ff = ff[(ff[:, 0] != ff[:, 1]) & (ff[:, 1] != ff[:, 2]) & (ff[:, 0] != ff[:, 2])]
    _, first = np.unique(np.sort(ff, 1), axis=0, return_index=True)
    ff = ff[np.sort(first)]
    n = int(inv.max()) + 1
    sums = np.zeros((n, 3))
    np.add.at(sums, inv, v)
    vv = sums / np.bincount(inv, minlength=n)[:, None]
    used = np.unique(ff)
    remap = -np.ones(len(vv), np.int64)
    remap[used] = np.arange(len(used))
    return vv[used].astype(np.float32), remap[ff].astype(np.uint32)


class MeshBank:
    def __init__(self):
        self.items, self.blobs, self.off, self.index = [], [], 0, {}

    def add(self, key, v, f):
        if key in self.index:
            return self.index[key]
        v, f = decimate(v, f)
        vb, fb = np.ascontiguousarray(v, np.float32).tobytes(), np.ascontiguousarray(f, np.uint32).tobytes()
        it = {"key": key, "v_off": self.off, "nv": len(v), "f_off": self.off + len(vb), "nf": len(f)}
        self.blobs += [vb, fb]
        self.off += len(vb) + len(fb)
        self.items.append(it)
        self.index[key] = len(self.items) - 1
        return self.index[key]

    def write(self, out):
        (out / "meshes.json").write_text(json.dumps(self.items))
        (out / "meshes.b64.txt").write_text(base64.b64encode(b"".join(self.blobs)).decode())


def write_bundle(out, name, arrays, meta):
    """<name>.json (meta + layout) and <name>.b64.txt (arrays back to back, 4-byte aligned)."""
    layout, blobs, off = {}, [], 0
    for k, a in arrays.items():
        a = np.asarray(a)
        dt = "int8" if a.dtype.kind in "biu" else "float16"
        b = np.ascontiguousarray(a.astype(np.int8 if dt == "int8" else np.float16)).tobytes()
        layout[k] = {"offset": off, "shape": list(a.shape), "dtype": dt}
        pad = (-len(b)) % 4
        blobs.append(b + b"\0" * pad)
        off += len(b) + pad
    (out / f"{name}.json").write_text(json.dumps({**meta, "name": name, "layout": layout}))
    (out / f"{name}.b64.txt").write_text(base64.b64encode(b"".join(blobs)).decode())


# ------------------------------------------------------------------ FK
def run_fk(task, ep, robot, trajs, joint_names, cache, refresh):
    if cache.exists() and not refresh:
        d = np.load(cache)
        if all(f"pose/{k}" in d.files and len(d[f"pose/{k}"]) == len(v) for k, v in trajs.items()):
            return d
    qfile = cache.with_suffix(".in.npz")
    np.savez(qfile, joint_names=np.array(joint_names), **{f"traj/{k}": v for k, v in trajs.items()})
    cmd = [os.environ.get("PY", sys.executable), str(HERE / "viewer_fk.py"), "--task", task, "--episode", str(ep),
           "--robot", robot, "--q", str(qfile), "--out", str(cache), "--headless", "--chunk", "64"]
    print(f"  FK {robot} ep{ep}: {', '.join(f'{k} {len(v)}' for k, v in trajs.items())}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0 or not cache.exists():
        sys.exit(f"viewer_fk failed for {robot}:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    qfile.unlink()
    return np.load(cache)


def robot_geoms(fk, bank, robot):
    """Links that have a visual mesh: (mesh index list, link names, link indices)."""
    mesh, link, idx = [], [], []
    for i, ln in enumerate(fk["link_names"]):
        if f"mesh/{i}/v" in fk.files:
            mesh.append(bank.add(f"{robot}:{ln}", fk[f"mesh/{i}/v"], fk[f"mesh/{i}/f"]))
            link.append(str(ln))
            idx.append(i)
    return {"mesh": mesh, "link": link}, idx


# ------------------------------------------------------------------ success chips
def short(obj_id):
    parts = obj_id.split("_")
    return "_".join(parts[2:]) if len(parts) > 2 and parts[0] == "obj" else obj_id


def cond_label(node):
    def find(n, t):
        if n.get("type") == t:
            return n
        for c in n.get("conditions", []):
            r = find(c, t)
            if r:
                return r
    ins = find(node, "object_inside")
    if ins:
        return f"{short(ins['object'])} → {short(ins['container'])}"
    t, o = node.get("type"), short(node.get("object", ""))
    return {"object_in_zone": f"{o} 위치", "object_upright": f"{o} 세움", "object_static": f"{o} 정지",
            "joint_position": f"{o} 관절", "object_on_top": f"{o} 위에 놓임"}.get(t, f"{o} {t}")


def condition_tracks(scene_yaml, obj_ids, obj_pose_wxyz, art, fps):
    """Each terminal sub-condition evaluated on every rollout frame -> (labels, (T,C) int8)."""
    cfg = yaml.safe_load(open(scene_yaml))
    raw = ((cfg.get("metrics") or {}).get("terminal") or {}).get("raw_condition")
    if not raw:
        return [], None
    kids = raw["conditions"] if raw.get("type") == "all" else [raw]
    paths.use_bench2dex()
    from success.condition_evaluator import evaluate_condition_tree   # Bench2Dex
    T = len(obj_pose_wxyz)
    p = obj_pose_wxyz
    vel = np.zeros((T, len(obj_ids), 3), np.float32)
    vel[1:] = (p[1:, :, :3] - p[:-1, :, :3]) * fps
    out = np.zeros((T, len(kids)), np.int8)
    ctxs = [{"dt": 1.0 / fps} for _ in kids]
    for t in range(T):
        states = {}
        for j, k in enumerate(obj_ids):
            q = p[t, j]
            st = {"pose_world": np.array([*q[:3], *q[4:7], q[3]], np.float32),
                  "lin_vel_world": vel[t, j], "ang_vel_world": np.zeros(3, np.float32)}
            if k in art:
                st["joint_names"], st["qpos"] = art[k]["names"], art[k]["q"][t]
                st["qvel"] = np.zeros_like(st["qpos"])
            states[k] = st
        for c, node in enumerate(kids):
            try:
                out[t, c] = bool(evaluate_condition_tree(node, states, ctxs[c]))
            except Exception:  # noqa: BLE001  (an evaluator needing data we do not have)
                out[t, c] = -1
    os.chdir(ROOT)
    return [cond_label(k) for k in kids], out


# ------------------------------------------------------------------ live progress (run.log)
def progress(run_log, T):
    """Last SPIDER attempt in progress: frame, object errors, drops/backtracks, hold-pass shift."""
    import re
    if not run_log.exists():
        return None
    txt = run_log.read_text(errors="ignore")
    start = txt.rfind("stage3_spider.py")
    if start < 0:
        return None
    seg = txt[start:]
    tag = re.search(r"--tag (\S+)", seg)
    p = {"tag": tag.group(1) if tag else "spider", "T": T, "frame": 0, "obj_err_cm": None, "seconds": None,
         "drops": [], "hold": None, "done": "RESULT stage3" in seg}
    for m in re.finditer(r"frame (\d+)/(\d+) obj err cm \[([^\]]*)\] success now (\w+) \((\d+)s", seg):
        p["frame"], p["seconds"] = int(m.group(1)), int(m.group(5))
        p["obj_err_cm"] = [round(float(x), 1) for x in m.group(3).split(",")]
    for m in re.finditer(r"drop of \['([^']+)'\] at frame (\d+): back to (\d+), retry (\d+)/(\d+)", seg):
        p["drops"].append({"obj": m.group(1), "frame": int(m.group(2)), "back": int(m.group(3)),
                           "retry": int(m.group(4)), "max": int(m.group(5))})
    m = re.search(r"hold pass: reached-vs-kinematic tip shift mean ([\d.]+) cm, p95 ([\d.]+) cm, worst frame (\d+)", seg)
    if m:
        p["hold"] = {"mean_cm": float(m.group(1)), "p95_cm": float(m.group(2)), "worst_frame": int(m.group(3))}
    if p["drops"]:
        p["frame"] = max(p["frame"], max(d["frame"] for d in p["drops"]))
    return p


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--targets", nargs="*", default=None, help="default: every target with stage-2 output")
    ap.add_argument("--out", default=None)
    ap.add_argument("--refresh_fk", action="store_true")
    a = ap.parse_args()

    scene = paths.task_dir(a.task).parent.name
    out = Path(a.out) if a.out else paths.RUNS / "viewer" / scene
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    scene_yaml = ROOT / "scenes" / f"{scene}.yaml"
    cfg = yaml.safe_load(open(scene_yaml))
    bank = MeshBank()
    summary = {"task": scene, "description": cfg.get("description", ""), "fps": 20, "stride": STRIDE,
               "source": None, "targets": [], "geoms": {}, "objects": [], "conditions": [], "episodes": []}
    geom_idx = {}

    for ep in a.episodes:
        d = paths.run_dir(a.task, ep)
        name = f"ep{ep:03d}"
        ref = np.load(d / "reference.npz")
        meta = json.loads(str(ref["meta"]))
        src = meta["source"]
        targets = a.targets or sorted(p.name for p in d.iterdir() if (p / "kinematic.npz").exists())
        summary["source"] = {"name": src, "label": HAND.get(src, src)}
        for t in targets:
            if t not in [x["name"] for x in summary["targets"]]:
                summary["targets"].append({"name": t, "label": HAND.get(t, t)})
        T = meta["n_frames"]
        ks = np.arange(0, T, STRIDE)

        # objects: one mesh per body (reference meshes.npz, body frames)
        om = np.load(d / "meshes.npz")
        objects = []
        for bi, body in enumerate(meta["bodies"]):
            oid = meta["obj_ids"][meta["body_obj"][bi]]
            if f"{bi}/v" not in om.files:
                continue
            base = oid.split("_")[1] if oid.startswith("obj_") else oid
            mi = bank.add(f"obj:{base}:{body.split('/')[-1]}", om[f"{bi}/v"], om[f"{bi}/f"])
            objects.append({"id": body, "obj": oid, "label": short(body.split("/")[0]) +
                            ("" if "/" not in body else "/" + body.split("/")[1]), "mesh": mi, "body": bi})
        if not summary["objects"]:
            summary["objects"] = [{k: o[k] for k in ("id", "label")} for o in objects]
        body_ids = [o["body"] for o in objects]
        q_to_x = lambda p: np.concatenate([p[..., :3], p[..., 4:7], p[..., 3:4]], -1)   # wxyz -> xyzw

        # source FK (reference animation)
        fk = run_fk(a.task, ep, src, {"kin": ref["src_q"][ks]}, [str(n) for n in ref["joint_names"]],
                    d / f"viewer_fk_{src}.npz", a.refresh_fk)
        if src not in summary["geoms"]:
            summary["geoms"][src], geom_idx[src] = robot_geoms(fk, bank, src)
        contact = ref["contact_body"][ks]
        body_to_obj = {o["body"]: j for j, o in enumerate(objects)}
        contact = np.vectorize(lambda b: body_to_obj.get(int(b), -1))(contact).astype(np.int8)
        write_bundle(data, name, {
            "src_pose": fk["pose/kin"][:, geom_idx[src]],
            "obj_pose": q_to_x(ref["body_pose"][ks][:, body_ids]),
            "src_tips": ref["src_tips"][ks], "contact": contact,
            "active": np.vectorize(lambda b: body_to_obj.get(int(b), -1))(ref["active"][ks]).astype(np.int8),
        }, {"n": len(ks), "frame_index": ks.tolist(), "table_z": meta["table_z"],
            "objects": [{"id": o["id"], "mesh": o["mesh"]} for o in objects]})

        segs = {s: [{"obj": seg["body"], "start": seg["start"], "end": seg["end"]} for seg in v]
                for s, v in meta["segments"].items()}
        ep_entry = {"name": name, "frames": T, "table_z": meta["table_z"], "segments": segs,
                    "source_file": Path(meta["source_episode"]).name, "targets": {}}

        for t in targets:
            td = d / t
            kin = np.load(td / "kinematic.npz")
            trajs = {"kin": kin["q"][ks]}
            runs = []
            for rp in sorted(td.glob("kinreplay_rollout.npz")) + sorted(td.glob("spider_a*_rollout.npz")) + sorted(td.glob("spider_rollout.npz")):
                tag = rp.name[: -len("_rollout.npz")]
                ro = np.load(rp)
                trajs[tag] = ro["q"]
                runs.append((tag, ro))
            fk = run_fk(a.task, ep, t, trajs, [str(n) for n in kin["joint_names"]], td / "viewer_fk.npz", a.refresh_fk)
            if t not in summary["geoms"]:
                summary["geoms"][t], geom_idx[t] = robot_geoms(fk, bank, t)
            err = np.linalg.norm(kin["tips"] - ref["src_tips"], axis=-1)            # (T,10) m
            write_bundle(data, f"{name}_{t}_kin", {
                "tgt_pose": fk["pose/kin"][:, geom_idx[t]], "tgt_tips": kin["tips"][ks],
                "tip_err_mm": 1000 * err[ks]}, {"n": len(ks), "frame_index": ks.tolist()})
            inc = ref["contact_body"] >= 0
            stats = {}
            for h, side in enumerate(("right", "left")):
                e, c = 1000 * err[:, 5 * h:5 * h + 5], inc[:, 5 * h:5 * h + 5]
                stats[side] = {"tip_err_mean_mm": float(e.mean()), "tip_err_p95_mm": float(np.quantile(e, 0.95)),
                               "tip_err_contact_mm": float(e[c].mean()) if c.any() else None,
                               "contact_frames": int(c.any(1).sum())}
            status = json.load(open(td / "status.json")) if (td / "status.json").exists() else {}
            entry = {"kin": stats, "kin_file": f"{name}_{t}_kin", "state": status.get("state", "running"),
                     "kin_report": json.loads(str(kin["report"])), "runs": [],
                     "progress": progress(td / "run.log", T)}
            for tag, ro in runs:
                res = json.loads(str(ro["result"]))
                rec_p = td / f"{tag}_record.json"
                rec = json.load(open(rec_p)) if rec_p.exists() else {}
                obj_ids = [str(x) for x in ro["obj_ids"]]
                art = {k[4:]: {"names": [], "q": ro[k]} for k in ro.files if k.startswith("art/")}
                labels, tracks = condition_tracks(scene_yaml, obj_ids, ro["obj"], art, 20)
                summary["conditions"] = summary["conditions"] or labels
                n = len(ro["q"])
                rollout_obj = ro["obj"][:, [obj_ids.index(o["obj"]) for o in objects]]
                arrays = {"tgt_pose": fk[f"pose/{tag}"][:, geom_idx[t]], "obj_pose": q_to_x(rollout_obj),
                          "err_cm": 100 * ro["err"], "success": ro["success"].astype(np.int8)}
                if tracks is not None:
                    arrays["cond"] = tracks
                write_bundle(data, f"{name}_{t}_{tag}", arrays, {"n": n, "frame_index": list(range(n))})
                fin = {labels[c]: bool(tracks[-1, c] == 1) for c in range(len(labels))} if tracks is not None else {}
                args = res.get("args", {})
                success = rec.get("success", res.get("task_success"))
                runs_note = []
                if res.get("aborted"):
                    runs_note.append(res["aborted"])
                if res.get("retries"):
                    runs_note.append("backtracking " + ", ".join(f"{k}:{v}" for k, v in res["retries"].items()))
                label = ("기구학 경로 물리 재생 (stage 2 관절 경로 그대로, SPIDER 없음)" if tag == "kinreplay" else
                         f"SPIDER 시도 {tag.split('_a')[-1] if '_a' in tag else 0} · seed {args.get('seed')} · "
                         f"{args.get('num_samples')} 샘플 · iters {args.get('iters')}")
                entry["runs"].append({
                    "file": f"{name}_{t}_{tag}", "tag": tag, "label": label,
                    "task_success": bool(success), "conditions": fin, "score": int(sum(fin.values())),
                    "n_conditions": len(fin), "stable_frame": res.get("scene_stable_success_frame"),
                    "minutes": res.get("minutes"), "obj_err_final_cm": res.get("obj_err_final_cm"),
                    "note": " · ".join(runs_note)})
            ep_entry["targets"][t] = entry
            print(f"  {name} {t}: kin + {len(runs)} SPIDER run(s)", flush=True)
        summary["episodes"] = [e for e in summary["episodes"] if e["name"] != name] + [ep_entry]

    bank.write(data)
    (data / "summary.json").write_text(json.dumps(summary))
    shutil.copy(HERE / "index.html", out / "index.html")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"RESULT viewer {scene}: {len(summary['episodes'])} episode(s), targets "
          f"{[t['name'] for t in summary['targets']]}, {size / 1e6:.1f} MB -> {out}", flush=True)


if __name__ == "__main__":
    main()
