"""Append the legacy DexMachina residual-RL rollouts (task 06, RH56DFX -> Shadow, ep000/ep001; prototype
results in results/legacy_06_shadow/viewer) to the Cross-Hand Retarget Viewer packs as extra Shadow runs.

  python tools/retarget/viewer/merge_dexmachina_legacy.py --site <dir with data/summary.json + data/epNNN.pack.*> --out <dir>

Legacy bundles: tgt_pose [n,64,7] (links of geoms.target), obj_pose/err_cm in sorted-id order, stride 1.
Converted to the viewer's sim-bundle format: tgt_pose in geoms.shadow link order, objects in the episode's
order, every 2nd frame, success = 0 and cond = -1 (no per-frame Bench2Dex check was recorded) except the
last frame, which carries the legacy final-state verdict.
"""
import argparse, base64, json, shutil
from pathlib import Path
import numpy as np

LEGACY = Path(__file__).resolve().parents[3] / "results/legacy_06_shadow/viewer/data"
COND_MAP = {"object_in_zone:bowl_1": "bowl_1 위치", "object_upright:bowl_1": "bowl_1 세움",
            "all:banana_2+bowl_1": "banana_2 → bowl_1", "all:apple_3+bowl_1": "apple_3 → bowl_1",
            "all:apple_4+bowl_1": "apple_4 → bowl_1"}


def load_legacy(name):
    meta = json.loads((LEGACY / f"{name}.json").read_text())
    buf = base64.b64decode((LEGACY / f"{name}.b64.txt").read_text())
    arr = {k: np.frombuffer(buf, dtype=d["dtype"], count=int(np.prod(d["shape"])), offset=d["offset"]).reshape(d["shape"])
           for k, d in meta["layout"].items()}
    return meta, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    site, out = Path(a.site), Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    summary = json.loads((site / "data/summary.json").read_text())
    legacy = json.loads((LEGACY / "summary.json").read_text())
    cur_links = summary["geoms"]["shadow"]["link"]
    leg_links = legacy["geoms"]["target"]["link"]
    link_idx = [leg_links.index(n) for n in cur_links]
    conds = summary["conditions"]

    for ep in ("ep000", "ep001"):
        pack = json.loads((site / f"data/{ep}.pack.json").read_text())
        buf = bytearray(base64.b64decode((site / f"data/{ep}.pack.b64.txt").read_text()))
        obj_ids = [o["id"] for o in pack["bundles"][ep]["objects"]]
        e = next(x for x in summary["episodes"] if x["name"] == ep)
        targets = json.loads(e["targets"]) if isinstance(e["targets"], str) else e["targets"]
        runs = targets["shadow"]["runs"]
        runs[:] = [r for r in runs if not r["tag"].startswith("dm_")]  # idempotent re-merge

        # frame check: the arm base must sit where the current viewer's shadow kinematic bundle puts it
        kin = pack["bundles"][f"{ep}_shadow_kin"]["layout"]["tgt_pose"]
        kin_pose = np.frombuffer(bytes(buf), np.float16, count=int(np.prod(kin["shape"])), offset=kin["offset"]).reshape(kin["shape"])
        _, r0 = load_legacy(f"{ep}_r0")
        d = float(np.abs(kin_pose[0, 0, :3].astype(np.float32) - r0["tgt_pose"][0, link_idx[0], :3].astype(np.float32)).max())
        print(f"{ep}: base link offset between legacy and current frame = {d * 100:.2f} cm")
        assert d < 0.02, "legacy rollouts are not in the viewer's world frame"

        entries = [(r, f"dm_{r['tag']}", f"④ DexMachina residual RL · {r['label'].replace('DexMachina ', '')}")
                   for r in legacy["rollouts"][ep] if r["tag"].startswith("r")]
        entries += [(s, f"dm_{s['stage']}", f"⑤ DexMachina 학습 중 · {s['label']}") for s in legacy["training"].get(ep, [])]
        for src, tag, label in entries:
            meta, arr = load_legacy(src["file"])
            leg_ids = meta["obj_ids"]
            oi = [leg_ids.index(i) for i in obj_ids]
            sl = slice(None, None, 2)
            fi = meta["frame_index"] if isinstance(meta["frame_index"], list) else json.loads(meta["frame_index"])
            frame_index = [int(x) for x in fi[sl]]
            n = len(frame_index)
            last = src.get("conditions", {})
            cond = np.full((n, len(conds)), -1, np.int8)
            for k, v in last.items():
                if COND_MAP.get(k) in conds:
                    cond[-1, conds.index(COND_MAP[k])] = int(bool(v))
            new = {"tgt_pose": arr["tgt_pose"][sl][:, link_idx].astype(np.float16),
                   "obj_pose": arr["obj_pose"][sl][:, oi].astype(np.float16),
                   "err_cm": arr["err_cm"][sl][:, oi].astype(np.float16),
                   "success": np.zeros(n, np.int8), "cond": cond}
            layout = {}
            for k, v in new.items():
                while len(buf) % 8:
                    buf.append(0)
                layout[k] = {"offset": len(buf), "shape": list(v.shape), "dtype": str(v.dtype)}
                buf += np.ascontiguousarray(v).tobytes()
            name = f"{ep}_shadow_{tag}"
            pack["bundles"][name] = {"n": n, "frame_index": frame_index, "name": name, "layout": layout}
            note = src.get("note", "")
            fin = src.get("obj_err_final_cm") or src.get("final_err_cm") or {}
            runs.append({"file": name, "tag": tag, "label": label, "task_success": bool(src.get("task_success", False)),
                         "conditions": {COND_MAP[k]: v for k, v in last.items() if k in COND_MAP},
                         "score": src.get("score", 0), "n_conditions": src.get("n_conditions", len(COND_MAP)),
                         "stable_frame": None, "minutes": None, "obj_err_final_cm": fin,
                         "note": ("이전 프로토타입(2026-10-05) · 마지막 프레임만 판정 기록, 'bowl_1 정지' 미측정"
                                  + (f" · {note}" if note else ""))})
        e["targets"] = json.dumps(targets) if isinstance(e["targets"], str) else targets
        (out / f"data/{ep}.pack.json").write_text(json.dumps(pack))
        (out / f"data/{ep}.pack.b64.txt").write_text(base64.b64encode(bytes(buf)).decode())
        print(f"{ep}: +{len(entries)} DexMachina runs, pack {len(buf) / 1e6:.1f} MB")
    (out / "data/summary.json").write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
