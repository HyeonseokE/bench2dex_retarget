import argparse, os, time
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--camera", action="store_true")
p.add_argument("--png", default="")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
t0 = time.time()
app = AppLauncher(a).app
print(f"[L4] AppLauncher up in {time.time() - t0:.1f}s", flush=True)

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

if a.camera:
    from isaaclab.sensors import CameraCfg

    @configclass
    class CamSceneCfg(SceneCfg):
        cam = CameraCfg(prim_path="{ENV_REGEX_NS}/Cam", update_period=0, height=240, width=320,
                        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=24.0))
    cfg = CamSceneCfg(num_envs=a.num_envs, env_spacing=1.0)
else:
    cfg = SceneCfg(num_envs=a.num_envs, env_spacing=1.0)

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=a.device))
scene = InteractiveScene(cfg)
sim.reset()
print(f"[L4] scene: {a.num_envs} envs on {sim.device}, physics dt {sim.get_physics_dt():.4f}", flush=True)
if a.camera:
    o = scene.env_origins
    scene["cam"].set_world_poses_from_view(o + torch.tensor([0.6, 0.6, 0.5], device=o.device),
                                           o + torch.tensor([0.0, 0.0, 0.05], device=o.device))

n = 240
t0 = time.time()
for _ in range(n):
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
torch.cuda.synchronize()
dt = time.time() - t0
z = scene["cube"].data.root_pos_w[:, 2]
print(f"[L4] {n} steps in {dt:.2f}s -> {n / dt:.1f} steps/s, {n * a.num_envs / dt:,.0f} env-steps/s", flush=True)
print(f"[L4] cube z: min {z.min():.3f} max {z.max():.3f} (expect ~0.05 everywhere)", flush=True)
ok = bool((z < 0.2).all())
print(f"[L4] GPU mem peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB (torch only)", flush=True)

if a.camera:
    rgb = scene["cam"].data.output["rgb"][0, ..., :3].cpu()
    mean, std = float(rgb.float().mean()), float(rgb.float().std())
    print(f"[L4c] rgb {tuple(rgb.shape)} mean {mean:.1f} std {std:.1f}", flush=True)
    if a.png:
        from PIL import Image
        Image.fromarray(rgb.numpy()).save(a.png)
        print(f"[L4c] saved {a.png}", flush=True)
    ok = ok and std > 2.0     # an all-black or flat frame means the renderer produced nothing
    print("L4C_OK" if ok else "L4C_BAD", flush=True)
else:
    print("L4_OK" if ok else "L4_BAD", flush=True)
app.close()
