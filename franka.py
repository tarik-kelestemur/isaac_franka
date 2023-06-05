from omni.isaac.kit import SimulationApp
sim_app = SimulationApp({"headless": True})
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.franka import Franka
from omni.isaac.franka.kinematics_solver import KinematicsSolver
from omni.isaac.core import World

import hydra
from omegaconf import DictConfig

class FrankaSim:
    def __init__(self, cfg: DictConfig) -> None:
        self._setup_scene(cfg.livestream)
        
    def _setup_scene(self, livestream) -> None:
        self.world = World(stage_units_in_meters=1.0)
        self.franka = self.world.scene.add(Franka(prim_path="/World/Franka", name="my_franka"))
        self.world.scene.add_default_ground_plane()
        self.world.reset()
        if livestream:
            sim_app.set_setting("/app/livestream/enabled", True)
            sim_app.set_setting("/app/window/drawMouse", True)
            sim_app.set_setting("/app/livestream/proto", "ws")
            sim_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
            sim_app.set_setting("/ngx/enabled", False)
            enable_extension("omni.kit.livestream.native")

    def run(self) -> None:
        while sim_app.is_running():
            sim_app.update()

        sim_app.close()

@hydra.main(version_base=None, config_path="config", config_name="franka")
def main(cfg: DictConfig):
    franka = FrankaSim(cfg)
    franka.run()

if __name__ == "__main__":
    main()
    