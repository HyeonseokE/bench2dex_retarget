import time
t0 = time.time()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print(f"[L3] Kit up in {time.time() - t0:.1f}s", flush=True)
import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
print("[L3] renderer:", carb.settings.get_settings().get("/renderer/active"), flush=True)
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
cube = world.scene.add(DynamicCuboid(prim_path="/World/cube", name="cube",
                                     position=np.array([0.0, 0.0, 1.0]), size=0.1))
world.reset()
for _ in range(240):
    world.step(render=True)
z = float(cube.get_world_pose()[0][2])
print(f"[L3] cube z after 240 steps: {z:.3f} (expect ~0.05)", flush=True)
print("L3_OK" if z < 0.2 else "L3_BAD: cube did not fall", flush=True)
app.close()
