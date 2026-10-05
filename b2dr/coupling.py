"""How a hand's non-actuated joints move, learned from that hand's own teleop episodes.

Teleop drives only ``target_joint_names``; the remaining finger joints follow mimic constraints in
the USD (r = 0.95-0.99 against an actuated joint in the demos) or stay still. The kinematic stage
must respect that, otherwise it plans finger shapes the hand cannot take. Every one of the five
hands is the source of at least two tasks, so each one's coupling is fitted on its own recordings:

    q_passive = a * q_master + b     master: the best-correlated actuated joint of the same finger
    q_passive = b                    if the joint does not move (std < 1e-3 rad)

The result is cached per robot in $B2DR_RUNS/coupling/<robot>.json.
"""

from __future__ import annotations

import json
import re

import h5py
import numpy as np

from . import paths, robots

FINGER_TOKENS = ("thumb", "index", "middle", "ring", "little", "pinky",
                 "finger1", "finger2", "finger3", "finger4", "finger5")


def _finger(name: str) -> str | None:
    low = name.lower()
    for t in FINGER_TOKENS:
        if t in low:
            return "little" if t == "pinky" else t
    m = re.match(r"(l_)?(ff|mf|rf|lf|th)j", low)                     # Shadow
    return m.group(2) if m else None


def source_episodes(robot_key: str, per_task: int = 5) -> list:
    """Episode files recorded with this robot (scans the first episode of every task)."""
    out = []
    for d in sorted(paths.DATASET.glob("*/origin-generalization")):
        files = sorted(d.glob("episode_??????.hdf5"))
        if not files:
            continue
        with h5py.File(files[0], "r") as h:
            if h["meta/robot_key"][()].decode() == robot_key:
                out += files[:per_task]
    return out


def fit(spec: robots.RobotSpec, joint_names: list[str], hand_joints: dict, actuated: dict) -> dict:
    cache = paths.RUNS / "coupling" / f"{spec.name}.json"
    if cache.exists():
        return json.load(open(cache))
    files = source_episodes(spec.key)
    passive = {s: [j for j in hand_joints[s] if j not in actuated[s] and j not in spec.locked] for s in robots.SIDES}
    rules = {}
    if any(passive.values()):
        if not files:
            raise FileNotFoundError(f"no {spec.key} episodes under {paths.DATASET} to fit its joint coupling")
        Q, names = [], None
        for f in files:
            with h5py.File(f, "r") as h:
                names = [x.decode() for x in h["robot/joint_names"][:]]
                Q.append(h["robot/qpos"][:])
        Q = np.concatenate(Q)
        col = {n: Q[:, names.index(n)] for n in names}
        for s in robots.SIDES:
            for j in passive[s]:
                q = col[j]
                if q.std() < 1e-3:
                    rules[j] = {"master": None, "a": 0.0, "b": float(q.mean())}
                    continue

                def corr(m):
                    return abs(np.corrcoef(q, col[m])[0, 1]) if col[m].std() > 1e-6 else 0.0
                # Same finger first: in the demos fingers often close together, so another finger
                # can correlate as well, but in a retargeted grasp they move independently. Only
                # when no same-finger joint explains it (Schunk's spreads, j3/j4) look further.
                fj = _finger(j)
                same = [m for m in actuated[s] if fj is not None and _finger(m) == fj]
                best, glob = (max(same, key=corr) if same else None), max(actuated[s], key=corr)
                if best is None or corr(glob) - corr(best) > 0.1:
                    best = glob
                a, b = np.polyfit(col[best], q, 1)
                rules[j] = {"master": best, "a": float(a), "b": float(b), "r": float(np.corrcoef(q, col[best])[0, 1])}
        # A hand the operator barely used (Schunk's left) gives no usable correlation; the two hands
        # are mirror builds, so borrow the other side's rule when it is clearly better.
        for j, r in list(rules.items()):
            mj, mirror = _mirror(j, names), None
            if mj in rules and rules[mj]["master"] is not None:
                mirror = rules[mj]
            if mirror is not None and abs(mirror.get("r", 0.0)) >= 0.8 > abs(r.get("r", 0.0)) and r["master"] is not None:
                rules[j] = {"master": _mirror(mirror["master"], names), "a": mirror["a"], "b": mirror["b"],
                            "r": mirror["r"], "mirrored_from": mj}
    out = {"robot": spec.name, "fitted_on": [str(f) for f in files], "rules": rules}
    cache.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(cache, "w"), indent=1)
    return out


def _mirror(name: str, names: list[str]) -> str | None:
    for a, b in (("right_", "left_"), ("left_", "right_")):
        if name.startswith(a) and name.replace(a, b, 1) in names:
            return name.replace(a, b, 1)
    if name.startswith("l_") and name[2:] in names:
        return name[2:]
    if "l_" + name in names:
        return "l_" + name
    return None


def matrix(joint_names: list[str], free: list[str], rules: dict, locked=()):
    """q = C @ x + d over all joints; x are the free joints (arm + actuated) in ``free`` order."""
    J, X = len(joint_names), len(free)
    C, d = np.zeros((J, X)), np.zeros(J)
    for i, n in enumerate(joint_names):
        if n in free:
            C[i, free.index(n)] = 1.0
        elif n in rules:
            r = rules[n]
            if r["master"] is not None:
                C[i, free.index(r["master"])] = r["a"]
            d[i] = r["b"]
        # locked / unknown joints stay at 0
    return C, d
