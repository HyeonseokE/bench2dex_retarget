"""Create the experiments/ tree: one folder per target task, one sub-folder per target hand.

  experiments/<scene>/README.md            task: description, source hand, target hands
  experiments/<scene>/<target>/README.md   this (task, target) pair: what it is, where its data is, notes
  experiments/<scene>/<target>/config.yaml settings stage 3 runs with (parsed from the code defaults)

Existing README.md files are kept (they hold notes); config.yaml is rewritten so it always matches the
code. Status tables (STATUS.md, results.csv) are added per pair by status_report.py.

  python tools/retarget/init_experiments.py
"""

import ast
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from retarget import paths  # noqa: E402

ROBOTS = ["rh56dfx", "rh5dg2", "shadow", "schunk", "wuji"]
HAND_NAME = {"rh56dfx": "Inspire RH56DFX", "rh5dg2": "Inspire RH5DG2", "shadow": "Shadow Hand",
             "schunk": "Schunk SVH", "wuji": "Wuji"}
SOURCE = {"06": "rh56dfx", "12": "rh56dfx", "42": "rh56dfx", "07": "rh5dg2", "34": "rh5dg2", "60": "rh5dg2",
          "43": "shadow", "76": "shadow", "08": "schunk", "44": "schunk", "21": "wuji", "27": "wuji"}


def argparse_defaults(py: Path) -> dict:
    """Defaults of every ap.add_argument("--x", ..., default=...) in a script, without running it."""
    out = {}
    for n in ast.walk(ast.parse(py.read_text())):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument" and n.args:
            name = getattr(n.args[0], "value", "")
            if not str(name).startswith("--"):
                continue
            kw = {k.arg: k.value for k in n.keywords}
            if "default" in kw:
                try:
                    out[name[2:]] = ast.literal_eval(kw["default"])
                except ValueError:
                    pass
            elif kw.get("action") is not None and getattr(kw["action"], "value", "") == "store_true":
                out[name[2:]] = False
    return out


def cfg_defaults(py: Path, cls: str) -> dict:
    """Annotated field defaults of a (config)class."""
    for n in ast.walk(ast.parse(py.read_text())):
        if isinstance(n, ast.ClassDef) and n.name == cls:
            out = {}
            for b in n.body:
                if isinstance(b, ast.AnnAssign) and b.value is not None:
                    try:
                        out[b.target.id] = ast.literal_eval(b.value)
                    except ValueError:
                        pass
            return out
    return {}


def main():
    exp = paths.EXPERIMENTS
    stage3 = argparse_defaults(HERE / "stage3_spider.py")
    for k in ("task", "episode", "target", "tag", "env_cfg", "max_frames", "seed", "no_hold_pass"):
        stage3.pop(k, None)
    env = cfg_defaults(ROOT / "retarget" / "spider_env.py", "SpiderEnvCfg")
    attempts = argparse_defaults(HERE / "run_target.py")
    for task, src in SOURCE.items():
        scene_yaml = sorted((ROOT / "scenes").glob(f"{task}_*.yaml"))[0]
        scene = scene_yaml.stem
        desc = (yaml.safe_load(open(scene_yaml)) or {}).get("description", "").strip()
        targets = [r for r in ROBOTS if r != src]
        d = exp / scene
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "README.md").exists():
            (d / "README.md").write_text(
                f"# {scene}\n\n{desc}\n\n"
                f"- Source (teleop demos): UR5 + {HAND_NAME[src]} (`{src}`), 50 episodes\n"
                f"- Targets: {', '.join(f'[{HAND_NAME[t]}]({t}/)' for t in targets)}\n"
                f"- Scene: `scenes/{scene}.yaml`; data under `results/{scene}/`\n")
        for t in targets:
            p = d / t
            p.mkdir(exist_ok=True)
            if not (p / "README.md").exists():
                (p / "README.md").write_text(
                    f"# {scene}: {HAND_NAME[src]} -> {HAND_NAME[t]}\n\n"
                    f"Retargeting of the {scene} teleop demos from UR5 + {HAND_NAME[src]} onto UR5 + {HAND_NAME[t]}.\n\n"
                    f"| file | content |\n|---|---|\n"
                    f"| `config.yaml` | stage-3 settings in force (from the code defaults; each attempt also stores its own in `spider_aK.json`) |\n"
                    f"| `STATUS.md`, `results.csv` | per-episode progress and verdicts (status_report.py) |\n\n"
                    f"Data: `results/{scene}/epNNN/{t}/` (reference, kinematic, SPIDER trace/rollout, run.log), "
                    f"episodes: `results/dataset/{t}/{scene}/`.\n\n## Notes\n\n")
            cfg = {"task": scene, "source": src, "target": t,
                   "stage3_spider": stage3, "spider_env": env,
                   "attempts": {"max_attempts": attempts.get("max_attempts"),
                                "schedule": "attempt K: seed K, num_samples 1024*(1+K//2), iters 5+K"},
                   "physics": "Bench2Dex robot/scene settings, unchanged"}
            with open(p / "config.yaml", "w") as f:
                f.write(f"# Generated by tools/retarget/init_experiments.py from the code defaults.\n")
                yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"experiments tree ready: {exp}")


if __name__ == "__main__":
    main()
