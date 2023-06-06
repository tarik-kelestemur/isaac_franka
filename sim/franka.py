from omni.isaac.kit import SimulationApp
sim_app = SimulationApp({"headless": False})
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.sensors.camera import Camera, PinholeCameraCfg
from omni.isaac.orbit.controllers.differential_inverse_kinematics import (
    DifferentialInverseKinematics,
    DifferentialInverseKinematicsCfg,
)
from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
import omni.isaac.orbit.utils.kit as kit_utils

from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.carb import set_carb_setting
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.markers import StaticMarker


import torch
import hydra
import numpy as np
from omegaconf import DictConfig


class FrankaSim:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        if cfg.livestream:
            self._setup_livestream()
        self.sim = SimulationContext(
            physics_dt=cfg.physics_dt, 
            rendering_dt=cfg.rendering_dt, 
            backend="torch", 
            device="cuda:0"
        )
        set_camera_view([3.5, 3.5, 3.5], [0.0, 0.0, 0.0])
        self._setup_scene()
        self._setup_robot()        
        self._setup_camera()

    def _setup_livestream(self) -> None:
        sim_app.set_setting("/app/livestream/enabled", True)
        sim_app.set_setting("/app/window/drawMouse", True)
        sim_app.set_setting("/app/livestream/proto", "ws")
        sim_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
        sim_app.set_setting("/ngx/enabled", False)
        enable_extension("omni.kit.livestream.native")

    def _setup_scene(self) -> None:
        if self.sim.get_physics_context().use_gpu_pipeline:
            self.sim.get_physics_context().enable_flatcache(True)
        set_carb_setting(self.sim._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)
        kit_utils.create_ground_plane("/World/defaultGroundPlane")

        prim_utils.create_prim(
            f"/World/Objects/Obj",
            "Cube",
            translation=np.array([0.6, 0.0, 1.3]),
            scale=(0.03, 0.03, 0.03),
            semantic_label="Cuba_Object",
        )
        self.goal_marker = StaticMarker("/Visuals/eef_goal", count=1, scale=(0.1, 0.1, 0.1))
        self.eef_marker = StaticMarker("/Visuals/eef_pose", count=1, scale=(0.1, 0.1, 0.1))
    
    def _setup_robot(self):
        robot_cfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
        robot_cfg.data_info.enable_jacobian = True
        robot_cfg.rigid_props.disable_gravity = True
        self.robot = SingleArmManipulator(cfg=robot_cfg)
        self.robot.spawn("/World/franka", translation=(0.0, 0.0, 1.0))
        self.sim.reset()
        self.robot.initialize("/World/franka")
        # TODO: move this to a config file
        home_dof_pos = torch.tensor([0.0, 0.0, 0.0, -torch.pi/2, 0.0, torch.pi/2, torch.pi/4, 0.0, 0.0], device=self.robot.device)
        home_dof_vel = torch.zeros((1, 9), device=self.robot.device)
        self.robot.set_dof_state(home_dof_pos, home_dof_vel)
        self.robot.reset_buffers()

        self.ik_cfg = DifferentialInverseKinematicsCfg(
            command_type="pose_abs",
            ik_method="dls",
            position_offset=self.robot.cfg.ee_info.pos_offset,
            rotation_offset=self.robot.cfg.ee_info.rot_offset,
        )
        self.ik_solver = DifferentialInverseKinematics(self.ik_cfg, self.cfg.num_envs, self.sim.device)
        self.ik_solver.initialize()
        self.ik_solver.reset_idx()
    
    def _setup_camera(self):
        camera_cfg = PinholeCameraCfg(
            sensor_tick=0,
            height=480,
            width=640,
            data_types=["rgb"],
            usd_params=PinholeCameraCfg.UsdCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
        )
        self.camera = Camera(cfg=camera_cfg, device="cuda")
        self.camera.spawn("/World/CameraSensor")
        self.camera.initialize()
        for _ in range(14):
            self.sim.render()
    
    def get_images(self):
        self.camera.update(dt=0.0)
        rgb = self.camera.data.output["rgb"]
        return rgb

    def move_eef(self, pos: torch.Tensor, rot: torch.Tensor):
        ik_cmd = torch.cat((pos, rot), dim=0).to(self.robot.device).unsqueeze(0)
        self.ik_solver.set_command(ik_cmd)
        q_sol = self.ik_solver.compute(
            self.robot.data.ee_state_b[:, 0:3],
            self.robot.data.ee_state_b[:, 3:7],
            self.robot.data.ee_jacobian,
            self.robot.data.arm_dof_pos,
        )
        joint_angle_cmds = torch.zeros((1, self.robot.num_actions), device=self.robot.device)
        joint_angle_cmds[0, : self.robot.arm_num_dof] = q_sol
        arm_command_offset = self.robot.data.actuator_pos_offset[:, : self.robot.arm_num_dof]
        joint_angle_cmds[:, : self.robot.arm_num_dof] -= arm_command_offset
        self.robot.apply_action(joint_angle_cmds)
        return q_sol 
    
    def run(self) -> None:
        n = 0
        goal_pos = torch.tensor([0.6, 0.0, 0.4])
        goal_rot = torch.tensor([0.0, 1.0, 0.0, 0.0])
        while sim_app.is_running():
            self.sim.step()
            self.robot.update_buffers(self.sim.get_physics_dt())
            self.goal_marker.set_world_poses(goal_pos.unsqueeze(0), goal_rot.unsqueeze(0))
            self.eef_marker.set_world_poses(self.robot.data.ee_state_w[:, 0:3], self.robot.data.ee_state_w[:, 3:7])
            if n == 50:
                q_sol = self.move_eef(pos=goal_pos, rot=goal_rot)
            n += 1
            print(f"timestep {n}")
        sim_app.close()

@hydra.main(version_base=None, config_path="config", config_name="franka")
def main(cfg: DictConfig):
    franka = FrankaSim(cfg)
    franka.run()

if __name__ == "__main__":
    main()
    