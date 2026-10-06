"""One-to-one audit of a retargeted replay episode against the released Bench2Dex data.

  python scripts/audit_episode.py --cand <our replay-generalization/episode.hdf5> \
      --task_ref <released replay of the same task> --robot_ref <released replay recorded with the target hand> \
      --source <released origin episode the retarget came from>

Every check prints OK / DIFF / NOTE with the evidence:
  structure   every dataset of each reference exists with the same dtype and shape (frame and joint axes
              normalised); task_ref decides task-specific paths, robot_ref hand-specific ones
  robot       joint_names / action_names / tactile sites + TacMap meta equal the target hand's release
  convention  Convention-A actions (frame 0 'none'), action_valid, sim_step stride, timestamps,
              episode flags, frame_valid/errors -- same patterns as the release
  meta        every released meta key present; values inherited from the source episode unchanged
              (scene, seeds, instruction, generalization sample, bboxes, cameras); retarget fields set
  cameras     ids, intrinsics, fisheye params equal the release; static-camera extrinsics equal the
              source episode's (restored generalization), wrist cameras follow the new hand
  objects     task objects + distractors present, unit quaternions, distractors = source poses
  labels      box3d for every object, box2d per camera, occupancy legend like the release
"""

import argparse
import json
import re

import h5py
import numpy as np

R = {"OK": 0, "DIFF": 0, "NOTE": 0}


def say(kind, msg):
    R[kind] += 1
    print(f"  [{kind:4s}] {msg}")


def dec(v):
    v = v[()] if isinstance(v, h5py.Dataset) else v
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray) and v.dtype.kind in "OS":
        return [x.decode() if isinstance(x, bytes) else x for x in v.tolist()]
    return v


def norm_path(p):
    p = re.sub(r"^metrics/timeseries/(.*?)_(obj|distractor)_.*$", r"metrics/timeseries/\1_<obj>", p)
    p = re.sub(r"^metrics/timeseries/stage_.*_(completed|current)$", r"metrics/timeseries/stage_<s>_\1", p)
    p = re.sub(r"^(objects|labels/box3d)/[^/]+", r"\1/<obj>", p)
    p = re.sub(r"^labels/box2d/([^/]+)/[^/]+", r"labels/box2d/\1/<obj>", p)
    p = re.sub(r"^robot/tactile/(tacmap|contact_mask|distance_along_normal_m)/[^/]+", r"robot/tactile/\1/<site>", p)
    return p


def tree(h):
    n = len(h["time/sim_step"])
    out = {}

    def v(name, o):
        if isinstance(o, h5py.Dataset):
            shape = tuple("T" if (i == 0 and s == n) else ("3T" if (i == 0 and abs(s - 3 * n) <= 3) else s)
                          for i, s in enumerate(o.shape))
            if name.startswith(("robot/q", "action/commanded", "action/action_names", "robot/joint_names")):
                shape = tuple(x if isinstance(x, str) else "J" for x in shape)
            if name.startswith("labels/occupancy_gt/") and len(shape) == 4:
                shape = (shape[0], "X", "Y", "Z")                     # per-episode auto bounds
            out.setdefault(norm_path(name), (shape, str(o.dtype)))
    h.visititems(v)
    return out


def structure(c, ref, label):
    tc, tr = tree(c), tree(ref)
    miss = sorted(k for k in tr if k not in tc and not k.startswith("meta/"))
    extra = sorted(k for k in tc if k not in tr and not k.startswith("meta/"))
    diff = sorted(k for k in tr if k in tc and tr[k] != tc[k] and not k.startswith("meta/"))
    data_paths = [k for k in tr if not k.startswith(("meta/", "metrics/"))]
    say("OK" if not [k for k in miss if not k.startswith("metrics/")] else "DIFF",
        f"vs {label}: {len(data_paths)} data paths; missing (non-metric) "
        f"{[k for k in miss if not k.startswith('metrics/')][:8]}")
    m_miss = [k for k in miss if k.startswith("metrics/")]
    m_extra = [k for k in extra if k.startswith("metrics/")]
    if m_miss or m_extra:
        say("NOTE", f"vs {label}: metric keys differ (MetricTracker version): missing {m_miss[:6]} extra {m_extra[:6]}")
    nm = [k for k in diff if not k.startswith("metrics/")]
    say("OK" if not nm else "DIFF", f"vs {label}: dtype/shape of shared data paths " + ("identical" if not nm else str([(k, tr[k], tc[k]) for k in nm[:6]])))
    md = [k for k in diff if k.startswith("metrics/")]
    if md:
        say("NOTE", f"vs {label}: metric dtype/shape differences {[(k, tr[k], tc[k]) for k in md[:4]]}")
    ne = [k for k in extra if not k.startswith("metrics/")]
    if ne:
        say("NOTE", f"vs {label}: extra data paths {ne[:8]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--task_ref", required=True)
    ap.add_argument("--robot_ref", required=True)
    ap.add_argument("--source", required=True)
    a = ap.parse_args()
    c, tref, rref, src = (h5py.File(p, "r") for p in (a.cand, a.task_ref, a.robot_ref, a.source))

    print("== structure")
    structure(c, tref, "task release")
    structure(c, rref, "robot release")

    print("== robot (target hand vs its own release)")
    for k in ("robot/joint_names", "action/action_names"):
        say("OK" if dec(c[k]) == dec(rref[k]) else "DIFF", f"{k}: {len(dec(c[k]))} names, " +
            ("identical order" if dec(c[k]) == dec(rref[k]) else "differs"))
    cs, rs = dec(c["robot/tactile/meta/site_names"]), dec(rref["robot/tactile/meta/site_names"])
    say("OK" if cs == rs else "DIFF", f"tactile sites {cs}")
    for k in rref["robot/tactile/meta"]:
        cv, rv = dec(c[f"robot/tactile/meta/{k}"]), dec(rref[f"robot/tactile/meta/{k}"])
        same = np.array_equal(np.asarray(cv), np.asarray(rv)) if not isinstance(cv, str) else cv == rv
        say("OK" if same else "DIFF", f"tactile meta/{k}: {cv!r}" + ("" if same else f" vs release {rv!r}"))
    say("OK" if dec(c["meta/robot_key"]) == dec(rref["meta/robot_key"]) else "DIFF", f"meta/robot_key {dec(c['meta/robot_key'])}")

    print("== convention (frames, actions, time, flags)")
    for h, name in ((c, "cand"), (rref, "robot release"), (tref, "task release")):
        src_ = dec(h["action/source"])
        av = h["action/action_valid"][:]
        cmd = h["action/commanded"][:]
        st = h["time/sim_step"][:]
        ts = h["time/timestamp_ns"][:]
        ep = {k: h[f"episode/{k}"][:] for k in ("is_first", "is_last", "done", "success")}
        print(f"    {name:14s} frames {len(st):4d} | source[0]={src_[0]!r} others={sorted(set(src_[1:]))} | "
              f"action_valid first/last {bool(av[0])}/{bool(av[-1])} invalid {int((~av).sum())} | NaN cmd rows {int(np.isnan(cmd).any(1).sum())} | "
              f"sim_step diffs {sorted(set(np.diff(st).tolist()))[:4]} | ts monotonic {bool((np.diff(ts) > 0).all())} | "
              f"is_first {np.where(ep['is_first'])[0].tolist()} is_last {np.where(ep['is_last'])[0].tolist()} "
              f"done {np.where(ep['done'])[0].tolist()} success {np.where(ep['success'])[0].tolist()} | "
              f"frame_valid {bool(h['frame_valid'][:].all())} errors {int(sum(1 for e in dec(h['frame_errors']) if e))}")
    say("OK" if dec(c["action/source"])[0] == "none" and not c["action/action_valid"][-1] else "DIFF",
        "Convention A: frame 0 has no action, last frame action_valid False")
    for k in ("action/control_mode", "action/action_type"):
        say("OK" if dec(c[k]) == dec(tref[k]) else "DIFF", f"{k} = {dec(c[k])!r} (release {dec(tref[k])!r})")

    print("== meta")
    rk, ck = set(tref["meta"]), set(c["meta"])
    say("OK" if rk <= ck else "DIFF", f"released meta keys present: {len(rk & ck)}/{len(rk)}; missing {sorted(rk - ck)}; added {sorted(ck - rk)}")
    inherit = ("scene_file", "scene_name", "task_name", "seed", "base_seed", "episode_seed", "task_base_seed",
               "task_seed_id", "seed_namespace", "seed_policy", "instruction", "fps", "physics_dt", "step_stride",
               "local_bboxes", "object_asset_keys", "object_asset_paths", "object_body_types", "object_roles",
               "camera_definitions", "pinhole_camera_ids", "fisheye_camera_ids", "box2d_camera_ids",
               "scene_generalization_config", "collect_config", "rgb_encoding", "rgb_jpeg_quality", "sensor_profile_id",
               "sensor_profile_version", "dataset_version", "schema_version", "task_manifest")
    bad = [k for k in inherit if k in src["meta"] and k in c["meta"] and dec(src["meta"][k]) != dec(c["meta"][k])]
    say("OK" if not bad else "DIFF", f"inherited from the source episode unchanged: {len(inherit) - len(bad)}/{len(inherit)} {('differ: ' + str(bad)) if bad else ''}")
    s_src = json.loads(dec(src["meta/scene_generalization_sample"]))
    s_c = json.loads(dec(c["meta/scene_generalization_sample"]))
    dk = [k for k in s_src if k not in ("robot_key", "embodiment", "_replay_generalization_mode") and s_src[k] != s_c.get(k)]
    say("OK" if not dk else "DIFF", f"scene_generalization_sample = source except robot/replay mode; differs in {dk}")
    print(f"    retarget fields: robot_key={dec(c['meta/robot_key'])} success={dec(c['meta/success'])} "
          f"collection_mode={dec(c['meta/collection_mode'])} has_rgb={dec(c['meta/has_rgb'])} has_tactile={dec(c['meta/has_tactile'])} "
          f"modalities={dec(c['meta/modalities'])} (release {dec(tref['meta/modalities'])})")

    print("== cameras")
    cams = sorted(c["cameras"])
    say("OK" if cams == sorted(tref["cameras"]) else "DIFF", f"camera ids {cams}")
    for cam in cams:
        g, r = c[f"cameras/{cam}"], tref[f"cameras/{cam}"]
        for k in r:
            if k in ("rgb", "extrinsic_world_from_cam"):
                continue
            same = np.allclose(g[k][()], r[k][()]) if r[k].dtype.kind in "fiub" else dec(g[k]) == dec(r[k])
            if not same:
                say("DIFF", f"{cam}/{k} differs from release")
        img = g["rgb"][0]
        say("OK", f"{cam}: {len(g['rgb'])} JPEG frames, first {len(img)} bytes; intrinsics/model = release") if True else None
        if "extrinsic_world_from_cam" in src[f"cameras/{cam}"] if f"cameras/{cam}" in src else False:
            e_c, e_s = g["extrinsic_world_from_cam"][0], src[f"cameras/{cam}/extrinsic_world_from_cam"][0]
            err = float(np.abs(e_c - e_s).max())
            kind = "wrist (moves with the new hand)" if "wrist" in cam else "static"
            say("OK" if ("wrist" in cam or err < 1e-4) else "DIFF", f"{cam} extrinsic frame 0 vs source episode: max |d| {err:.2e} [{kind}]")

    print("== objects")
    oc, osrc = sorted(c["objects"]), sorted(src["objects"])
    say("OK" if oc == osrc else "DIFF", f"objects {oc}")
    for k in oc:
        q = c[f"objects/{k}/pose_world"][:, 3:7]
        nq = float(np.abs(np.linalg.norm(q, axis=1) - 1).max())
        line = f"{k}: unit quats (max |1-|q||={nq:.1e})"
        if k.startswith("distractor"):
            n = min(len(q), len(src[f"objects/{k}/pose_world"]))
            dd = float(np.abs(c[f"objects/{k}/pose_world"][:n] - src[f"objects/{k}/pose_world"][:n]).max())
            line += f", = source poses (max |d| {dd:.1e})"
        say("OK" if nq < 1e-3 else "DIFF", line)

    print("== labels")
    b3 = sorted(c["labels/box3d"])
    say("OK" if b3 == oc else "DIFF", f"box3d for {len(b3)}/{len(oc)} objects")
    for cam in sorted(c["labels/box2d"]):
        vis = {k: int(c[f"labels/box2d/{cam}/{k}/visible"][:].sum()) for k in c[f"labels/box2d/{cam}"]}
        print(f"    box2d {cam}: visible frames {vis}")
    lc, lr = dec(c["labels/occupancy_gt/semantic_legend"]), dec(tref["labels/occupancy_gt/semantic_legend"])
    say("OK", f"occupancy voxel {dec(c['labels/occupancy_gt/voxel_size'])} grid {c['labels/occupancy_gt/state'].shape[1:]} (release {tref['labels/occupancy_gt/state'].shape[1:]}); legend {'same' if lc == lr else 'differs'}")

    print(f"== summary: OK {R['OK']}  DIFF {R['DIFF']}  NOTE {R['NOTE']}")


if __name__ == "__main__":
    main()
