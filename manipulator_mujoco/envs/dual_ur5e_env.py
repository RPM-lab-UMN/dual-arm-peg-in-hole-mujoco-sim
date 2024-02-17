import time
import os
import numpy as np
from dm_control import mjcf
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from manipulator_mujoco.arenas import StandardArena
from manipulator_mujoco.robots import Arm, Robotiq_2F85
from manipulator_mujoco.mocaps import Target
from manipulator_mujoco.props import Primitive, PegHole
from manipulator_mujoco.controllers import OperationalSpaceController

class DualUR5eEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }  # TODO add functionality to render_fps

    def __init__(self, render_mode=None):
        # TODO come up with an observation space that makes sense
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
        )

        # TODO come up with an action space that makes sense
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(6,), dtype=np.float64
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode

        ############################
        # create MJCF model
        ############################
        
        # checkerboard floor
        self._arena = StandardArena()

        # mocap target that OSC will try to follow
        self._left_target = Target(self._arena.mjcf_model, name="left_mocap")
        self._right_target = Target(self._arena.mjcf_model, name="right_mocap")

        # ur5e arm
        self._left_arm = Arm(
            xml_path= os.path.join(
                os.path.dirname(__file__),
                '../assets/robots/ur5e/ur5e.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site'
        )

        self._right_arm = Arm(
            xml_path= os.path.join(
                os.path.dirname(__file__),
                '../assets/robots/ur5e/ur5e.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site'
        )

        # gripper
        self._left_gripper = Robotiq_2F85()
        self._right_gripper = Robotiq_2F85()

        # small box to be manipulated
        self._box = Primitive(type="box", size=[0.01, 0.01, 0.01], pos=[0,0,0.02], rgba=[1, 0, 0, 1], friction=[1, 0.3, 0.0001])
        self._peg = PegHole(ph_type="peg", shape="Circle")
        self._hole = PegHole(ph_type="hole", shape="Circle")

        # attach gripper to arm
        self._left_arm.attach_tool(self._left_gripper.mjcf_model, pos=[0, 0, 0], quat=[0, 0, 0, 1])
        self._right_arm.attach_tool(self._right_gripper.mjcf_model, pos=[0, 0, 0], quat=[0, 0, 0, 1])

        # attach arm to arena
        self._arena.attach(
            self._left_arm.mjcf_model, pos=[-0.5,0,0.5], quat=[-0.5, -0.5, -0.5, 0.5]
        )

        self._arena.attach(
            self._right_arm.mjcf_model, pos=[0.5,0,0.5], quat=[-0.5, -0.5, -0.5, 0.5]
        )

        # attach box to arena as free joint
        # self._arena.attach_free(
        #     self._box.mjcf_model, pos=[-0.5,0.1,0.5]
        # )

        # attach peg to arena as free joint
        self._arena.attach_free(
            self._peg.mjcf_model, pos=[-0.5,0.1,0.2], quat=[0,0,-0.7071081,0.7071055]
        )

        self._arena.attach_free(
            self._hole.mjcf_model, pos=[-0.5,0.1,0], quat=[0,0,0.7071081,0.7071055]
        )
       
        # generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # set up OSC controller
        self._left_controller = OperationalSpaceController(
            physics=self._physics,
            joints=self._left_arm.joints,
            eef_site=self._left_arm.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=200,
            ko=200,
            kv=50,
            vmax_xyz=1.0,
            vmax_abg=2.0,
        )

        self._right_controller = OperationalSpaceController(
            physics=self._physics,
            joints=self._right_arm.joints,
            eef_site=self._right_arm.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=200,
            ko=200,
            kv=50,
            vmax_xyz=1.0,
            vmax_abg=2.0,
        )

        # for GUI and time keeping
        self._timestep = self._physics.model.opt.timestep
        self._viewer = None
        self._step_start = None


    def _get_obs(self) -> np.ndarray:
        # TODO come up with an observations that makes sense for your RL task
        return np.zeros(6)

    def _get_info(self) -> dict:
        # TODO come up with an info dict that makes sense for your RL task
        return {}

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)

        # reset physics
        with self._physics.reset_context():
            # put arm in a reasonable starting position

            self._physics.bind(self._left_arm.joints).qpos = [
                0.0,
                3.1415,
                0.0,
                0.0, # 
                0.0,
                0.0,
            ]
            # [
            #     0.0,
            #     -1.5707,
            #     1.5707,
            #     -1.5707,
            #     -1.5707,
            #     0.0,
            # ]

            self._physics.bind(self._right_arm.joints).qpos = [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]

            # put target in a reasonable starting position
            self._left_target.set_mocap_pose(self._physics, position=[-0.3, -0.4, 0.7], quaternion=[0, -0.70710677, 0, 0.70710677])
            self._right_target.set_mocap_pose(self._physics, position=[0.3, -0.4, 0.7], quaternion=[0, 0.70710677, 0, 0.70710677])

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        # TODO use the action to control the arm

        # get mocap target pose
        left_target_pose = self._left_target.get_mocap_pose(self._physics)
        right_target_pose = self._right_target.get_mocap_pose(self._physics)

        # run OSC controller to move to target pose
        self._left_controller.run(left_target_pose)
        self._right_controller.run(right_target_pose)

        # step physics
        self._physics.step()

        # render frame
        if self._render_mode == "human":
            self._render_frame()
        
        # TODO come up with a reward, termination function that makes sense for your RL task
        observation = self._get_obs()
        reward = 0
        terminated = False
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self) -> np.ndarray:
        """
        Renders the current frame and returns it as an RGB array if the render mode is set to "rgb_array".

        Returns:
            np.ndarray: RGB array of the current frame.
        """
        if self._render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> None:
        """
        Renders the current frame and updates the viewer if the render mode is set to "human".
        """
        if self._viewer is None and self._render_mode == "human":
            # launch viewer
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
            )
        if self._step_start is None and self._render_mode == "human":
            # initialize step timer
            self._step_start = time.time()

        if self._render_mode == "human":
            # render viewer
            self._viewer.sync()

            # TODO come up with a better frame rate keeping strategy
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            self._step_start = time.time()

        else:  # rgb_array
            return self._physics.render()

    def close(self) -> None:
        """
        Closes the viewer if it's open.
        """
        if self._viewer is not None:
            self._viewer.close()