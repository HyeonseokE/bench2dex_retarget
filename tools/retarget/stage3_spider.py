"""Stage 3: SPIDER physics retargeting (Meta FAIR, arXiv 2511.09484) of one episode onto one robot.

Port of the task-06 prototype (retarget/spider_retarget.py, SPIDER v2 settings) to every task and
robot. No learning: a receding-horizon sampling optimiser (MPPI) refines the kinematic reference
so the target robot reproduces the demo object motion under Bench2Dex physics.

  * Every replanning step the current state is broadcast to all N envs; N residual sequences
    (knots every `knot` frames over `horizon` frames, linearly interpolated) are rolled out and
    scored; the nominal plan is the cost-weighted mean (MPPI), `iters` times.
  * Virtual contact guidance pulls demo contact points towards the fingertips during
    optimisation, annealed to 0; committed frames always run without it.
  * Only frames near manipulation are optimised (gate); elsewhere the reference is replayed.
  * Only when manipulation is within `gate_look` frames ahead does a replanning step optimise.
  * Backtracking: when a manipulated object jumps away from its reference, roll back and re-plan the
    segment with more effort. Distance back from the segment's first drop: `back_frames` twice, then
    +2 frames per retry up to `back_max` (10, 10, 12, 14, 16, 18, 20). If the segment still fails, the
    run continues (an object is often re-acquired later); `--abort_on_fail 1` stops the episode instead.
  * A GPU physics failure (e.g. CUDA OOM: PhysX then returns frozen/identical object states) stops
    the run with an error instead of producing a fake success.
  * After the demo the last command is held for `settle` frames, as a Bench2Dex rollout keeps
    simulating.
  * The committed execution (env 0) is kept as a state trace: the state after every physics step
    and the joint target executed at every control step (retarget.recorder.capture), truncated with
    the rollout on backtracking. Samples are never recorded. Stage 4 writes the trace as a
    Bench2Dex episode; re-executing the joint targets open-loop does NOT reproduce the run (every
    commit starts from a restored snapshot), so the states are the record.
  * Success reported here is the scene's success_conditions with the dwell rule, a progress
    signal; the official verdict is MetricTracker's, computed by stage 4 on the trace.

  python tools/retarget/stage3_spider.py --task 06 --episode 0 --target shadow --headless
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--episode", type=int, required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--num_samples", type=int, default=1024)
ap.add_argument("--horizon", type=int, default=30)
ap.add_argument("--knot", type=int, default=5)
ap.add_argument("--commit", type=int, default=5)
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--sigma", type=float, default=0.5)
ap.add_argument("--guide_gain", type=float, default=300.0, help="N/m, contact spring at the first iteration")
ap.add_argument("--squeeze", type=float, default=0.02, help="m; contact targets moved this far into the body")
ap.add_argument("--retry_mode", choices=["schedule", "legacy"], default="schedule",
                help="schedule: per segment 10,10,12..20 frames back; legacy: the prototype's 2 commits back, 3 per frame")
ap.add_argument("--back_frames", type=int, default=10, help="first backtrack distance (frames), tried twice")
ap.add_argument("--back_max", type=int, default=20, help="largest backtrack distance (frames), +2 per further retry")
ap.add_argument("--gate_look", type=int, default=26, help="optimise only when manipulation is this many frames ahead")
ap.add_argument("--abort_on_fail", type=int, default=0,
                help="stop the episode when a segment fails after all retries. Off: the 06 prototype's successes "
                     "often recovered an object after 6-18 drops; aborting at 5-7 failed every episode it hit")
ap.add_argument("--drop_jump", type=float, default=0.025, help="m; error increase per commit counted as a drop")
ap.add_argument("--settle", type=int, default=40)
ap.add_argument("--env_cfg", default="", help="JSON overrides of SpiderEnvCfg fields")
ap.add_argument("--no_hold_pass", action="store_true")
ap.add_argument("--max_frames", type=int, default=0, help="debug: stop the episode early (result not saved as final)")
ap.add_argument("--tag", default="spider", help="output name: <tag>.json / <tag>_rollout.npz")
ap.add_argument("--seed", type=int, default=0, help="MPPI sampling seed (recorded in the result)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # Bench2Dex root
from retarget import paths, recorder  # noqa: E402
from retarget.spider_env import SpiderEnv, SuccessTracker, make_cfg  # noqa: E402

ART_W = 10.0      # cost per rad (or m) of articulation joint error, same weight as object position


def step_cost(env):
    """Per-env cost of the current state (lower is better)."""
    f = env.f()
    roots, _ = env.object_roots()
    ref = env.ref_root[f]
    c = (10.0 * (ref[..., :3] - roots[..., :3]).norm(dim=-1)
         + 0.5 * env.obj_rot_dist(roots[..., 3:7], ref[..., 3:7])).sum(-1)
    for k, v in env.object_qpos().items():
        c = c + ART_W * (env.ref_art[k][f] - v).abs().sum(-1)
    body = env.object_body_poses()
    ctgt, valid, idx = env.demo_contact_world(body, f)
    if args.squeeze > 0:   # demo contacts are surface points ~1-2 cm off the source pads: aim inside
        centre = torch.gather(body[..., :3], 1, idx[..., None].expand(-1, -1, 3))
        inward = centre - ctgt
        ctgt = ctgt + args.squeeze * inward / inward.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    tips = env.tips()
    cd = (ctgt - tips).norm(dim=-1)
    v = valid.float()
    c = c + 5.0 * (cd * v).sum(-1) / v.sum(-1).clamp(min=1)
    c = c + 1.0 * (env.ref_tips[f] - tips).norm(dim=-1).mean(-1)
    return torch.nan_to_num(c, nan=1e3, posinf=1e3)


def obj_err(env):
    """(n_obj,) env-0 object error: root position (m), plus summed joint error for articulations."""
    f = env.f()
    roots, _ = env.object_roots()
    e = (env.ref_root[f][0, :, :3] - roots[0, :, :3]).norm(dim=-1)
    for k, v in env.object_qpos().items():
        j = env.obj_ids.index(k)
        e[j] = e[j] + (env.ref_art[k][f][0] - v[0]).abs().sum()
    return e


def knots_to_actions(knots, H, knot):
    N, K, A = knots.shape
    t = torch.arange(H, device=knots.device, dtype=torch.float32) / knot
    i0 = t.floor().long().clamp(max=K - 1)
    i1 = (i0 + 1).clamp(max=K - 1)
    w = (t - i0.float())[None, :, None]
    return knots[:, i0] * (1 - w) + knots[:, i1] * w


def hold_pass(env):
    """Reference post-processing: PD-track the kinematic reference with objects held on the demo."""
    env.reset()
    env.hold, env.guide_gain = True, 0.0
    zero = torch.zeros(env.num_envs, env.n_act, device=env.device)
    q, tips, wrist = [env.robot.data.joint_pos[0].clone()], [env.tips()[0].clone()], [env.body_pose(env.wrist_ids)[0].clone()]
    for _ in range(env.T - 1):
        env.step(zero)
        q.append(env.robot.data.joint_pos[0].clone())
        tips.append(env.tips()[0].clone())
        wrist.append(env.body_pose(env.wrist_ids)[0].clone())
    env.hold = False
    q, tips, wrist = torch.stack(q), torch.stack(tips), torch.stack(wrist)
    T = len(q)
    shift = (tips - env.ref_tips[:T]).norm(dim=-1)
    worst = int(shift.max(1).values.argmax())
    print(f"  hold pass: reached-vs-kinematic tip shift mean {100 * shift.mean():.2f} cm, "
          f"p95 {100 * shift.quantile(0.95):.2f} cm, worst frame {worst} "
          f"({np.round(100 * shift[worst].cpu().numpy(), 1).tolist()} cm per tip); "
          f"per-50-frame means {np.round(100 * shift.mean(1).cpu().numpy()[::50], 1).tolist()}", flush=True)
    env.ref_q[:T], env.ref_tips[:T], env.ref_wrist[:T] = q, tips, wrist


def main():
    out_dir = paths.run_dir(args.task, args.episode) / args.target
    res_path = out_dir / f"{args.tag}.json"
    if res_path.exists():
        print(f"skip: {res_path} exists", flush=True)
        return
    overrides = json.loads(args.env_cfg) if args.env_cfg else {}
    env = SpiderEnv(make_cfg(args.task, args.episode, args.target, args.num_samples, args.device, **overrides))
    if args.max_frames:
        env.T = min(env.T, args.max_frames)
    T, dev, A = env.T, env.device, env.n_act
    if not args.no_hold_pass:
        hold_pass(env)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.reset()
    tracker = SuccessTracker(env.scene_def, env.step_dt)
    H, knot, S = args.horizon, args.knot, args.commit
    K = H // knot + 1
    nominal = torch.zeros(K, A, device=dev)
    sigma = torch.full((A,), args.sigma, device=dev)
    rec = {"q": [], "q_target": [], "obj": [], "art": [], "err": []}
    t0, n_opt = time.time(), 0
    history, retries, level, level_until = [], {}, 0, 0
    fhist, first_drop, aborted = [], {}, ""           # per-frame snapshots for fine backtracking
    sched = [args.back_frames] * 2 + list(range(args.back_frames + 2, args.back_max + 1, 2))
    body_obj = env.body_obj.cpu().numpy()

    def record(cmd):
        roots, _ = env.object_roots()
        rec["q"].append(env.robot.data.joint_pos[0].cpu().numpy())
        rec["q_target"].append(cmd)
        rec["obj"].append(roots[0].cpu().numpy())
        rec["art"].append({k: v[0].cpu().numpy() for k, v in env.object_qpos().items()})
        rec["err"].append(obj_err(env).cpu().numpy())
        tracker.update(env.success_states(0))

    trace = []                                       # aligned with rec: one entry per frame

    def commit_step(a):
        subs = []
        cmd = env.commit_step(a, after_physics_step=lambda: subs.append(recorder.capture(env)))
        trace[-1]["cmd"] = cmd                       # executed from the previous frame on
        trace.append({"state": subs[-1], "substeps": subs, "cmd": None})
        record(cmd)

    trace.append({"state": recorder.capture(env), "substeps": [], "cmd": None})
    record(env.q_target[0].cpu().numpy())            # frame 0: the reset state
    while int(env.frame[0]) < T - 3:
        f = int(env.frame[0])
        history.append((env.snapshot(), len(rec["err"]), nominal.clone(), tracker.state()))
        if not fhist or fhist[-1][0]["frame"] != f:
            fhist.append(history[-1])
        h = min(H, T - 3 - f)
        look = min(h, args.gate_look) if args.gate_look > 0 else h
        gate = env.gate[f + 1:f + 1 + look].max().item() if h > 0 else 0.0
        if gate > 0:
            s = env.snapshot()
            iters = args.iters * (1 + level)
            for it in range(iters):
                env.guide_gain = args.guide_gain * (1.0 - it / max(iters - 1, 1))
                env.restore(s)
                eps = torch.randn(env.num_envs, K, A, device=dev) * sigma * (1.0 + 0.5 * level) * (1.0 - 0.6 * it / iters)
                eps[0] = 0.0
                knots = (nominal[None] + eps).clamp(-1.0, 1.0)
                acts = knots_to_actions(knots, h, knot)
                cost = torch.zeros(env.num_envs, device=dev)
                for k in range(h):
                    env.step(acts[:, k])
                    cost += step_cost(env) * (1.0 + 2.0 * (k == h - 1))
                cost += 0.05 * (knots ** 2).sum((1, 2))
                adv = cost - cost.min()
                w = torch.softmax(-adv / (0.1 * adv.mean().clamp(min=1e-6)), 0)
                nominal = (w[:, None, None] * knots).sum(0)
            n_opt += 1
            env.guide_gain = 0.0
            env.restore(s)
            commit = knots_to_actions(nominal[None], min(S, h), knot)[0]
        else:
            commit = torch.zeros(min(S, max(h, 1)), A, device=dev)
        for k in range(commit.shape[0]):
            commit_step(commit[k])
            fhist.append((env.snapshot(), len(rec["err"]), nominal.clone(), tracker.state()))
        del fhist[:-60]
        # GPU physics failure guard: after e.g. a CUDA OOM PhysX returns frozen, identical object states
        P = torch.as_tensor(np.array(rec["obj"][-1]))[:, :3]
        if not torch.isfinite(P).all() or (len(P) > 1 and (torch.pdist(P) < 1e-4).any()):
            raise RuntimeError(f"PhysX state corrupted at frame {int(env.frame[0])} (GPU error?) - result discarded")
        # backtracking: a manipulated object jumped away from its reference
        fa = min(int(env.frame[0]), T - 1)
        active = {int(body_obj[a]) for a in env.active[max(0, fa - 5):fa + 1].ravel() if a >= 0}
        prev_len = history[-1][1]
        if active and prev_len > 0:
            e_now, e_prev = rec["err"][-1], rec["err"][prev_len - 1]
            dropped = [o for o in active if e_now[o] > 0.04 and e_now[o] - e_prev[o] > args.drop_jump]
            if dropped and args.retry_mode == "legacy":
                # the 06 prototype's rule (its first 20 episodes: 14 success): go back 2 commits, up to 3 retries
                # per restart frame, then carry on; a later drop restarts from a new frame with a new budget
                back = max(len(history) - 2, 0)
                key = history[back][0]["frame"]
                if retries.get(key, 0) < 3:
                    retries[key] = level = retries.get(key, 0) + 1
                    snap, n_keep, nominal, tstate = history[back]
                    nominal = nominal.clone()
                    del history[back:]
                    del fhist[[i for i, x in enumerate(fhist) if x[0]["frame"] <= key][-1] + 1:]
                    for kk in rec:
                        del rec[kk][n_keep:]
                    del trace[n_keep:]
                    trace[-1]["cmd"] = None
                    tracker.load(tstate)
                    env.restore(snap)
                    level_until = key + 3 * S + H
                    print(f"  drop of {[env.obj_ids[x] for x in dropped]} at frame {fa}: back to {key}, "
                          f"retry {level}/3 with {args.iters * (1 + level)} iterations", flush=True)
                    continue
                dropped = []
            if dropped:
                o = dropped[0]
                seg = fa                                   # start of this object's manipulation segment
                while seg > 0 and (body_obj[env.active[seg - 1][env.active[seg - 1] >= 0]] == o).any():
                    seg -= 1
                rk = (o, seg)
                if retries.get(rk, 0) < len(sched):
                    retries[rk] = r = retries.get(rk, 0) + 1
                    first_drop.setdefault(rk, fa)
                    target = first_drop[rk] - sched[r - 1]
                    level = min(r, 4)                      # at most 5x iterations
                    cand = [i for i, x in enumerate(fhist) if x[0]["frame"] <= target]
                    bi = cand[-1] if cand else 0
                    snap, n_keep, nominal, tstate = fhist[bi]
                    nominal = nominal.clone()
                    key = snap["frame"]
                    del fhist[bi + 1:]
                    history = [x for x in history if x[0]["frame"] < key]
                    for kk in rec:
                        del rec[kk][n_keep:]
                    del trace[n_keep:]
                    trace[-1]["cmd"] = None
                    tracker.load(tstate)
                    env.restore(snap)
                    level_until = fa + S + H
                    print(f"  drop of {[env.obj_ids[x] for x in dropped]} at frame {fa}: back to {key}, "
                          f"retry {r}/{len(sched)} with {args.iters * (1 + level)} iterations", flush=True)
                    continue
                if args.abort_on_fail:
                    aborted = f"{env.obj_ids[o]} lost at frame {fa} after {retries[rk]} retries"
                    print(f"  ABORT: {aborted}", flush=True)
                    break
        if int(env.frame[0]) >= level_until:
            level = 0
        shift = max(commit.shape[0] // knot, 1)
        nominal = torch.cat([nominal[shift:], torch.zeros(shift, A, device=dev)], 0)
        if f // 50 != int(env.frame[0]) // 50:
            print(f"  frame {int(env.frame[0])}/{T} obj err cm {np.round(100 * rec['err'][-1], 1).tolist()} "
                  f"success now {tracker.history[-1]} ({time.time() - t0:.0f}s, {n_opt} optimised)", flush=True)

    # settle: hold the last command, no residual, no guidance
    env.guide_gain = 0.0
    for _ in range(0 if aborted else args.settle):
        commit_step(None)

    stable = tracker.stable_frame()
    er = np.array(rec["err"])
    res = {"task": env.meta["task"], "episode": args.episode, "source": env.meta["source"], "target": args.target,
           "policy": "spider", "frames": len(er), "settle_frames": args.settle, "seed": args.seed,
           "scene_success": stable is not None, "scene_stable_success_frame": stable,
           "instant_success_frames": int(sum(tracker.history)),
           "minutes": round((time.time() - t0) / 60, 1), "optimised_steps": n_opt,
           "retries": {str(k): v for k, v in retries.items()}, "aborted": aborted, "args": vars(args), "env_cfg": overrides,
           "obj_err_mean_cm": dict(zip(env.obj_ids, (100 * er.mean(0)).round(2).tolist())),
           "obj_err_final_cm": dict(zip(env.obj_ids, (100 * er[-1]).round(2).tolist()))}
    res["task_success"] = res["scene_success"]
    out_dir.mkdir(parents=True, exist_ok=True)
    art = {f"art/{k}": np.array([a[k] for a in rec["art"]]) for k in env.art_ids}
    recorder.save_trace(out_dir / f"{args.tag}_trace.pkl.gz", recorder.trace_header(env), trace)
    np.savez_compressed(out_dir / f"{args.tag}_rollout.npz", q=np.array(rec["q"]), cmd=np.array(rec["q_target"][1:]),
                        obj=np.array(rec["obj"]), err=er, success=np.array(tracker.history),
                        joint_names=np.array(env.joint_names), obj_ids=np.array(env.obj_ids),
                        result=np.array(json.dumps(res)), **art)
    json.dump(res, open(res_path, "w"), indent=1)
    print(f"RESULT stage3 {res['task']} ep{args.episode} {res['source']}->{args.target}: "
          f"{'SUCCESS' if res['task_success'] else 'fail'} (scene conditions, {res['minutes']} min)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
