import manipulator_mujoco
import gymnasium as gym

from manipulator_mujoco.utils import Demo, DemoRecorder, DemoScheduler, PegInHoleDemo

def main():
    render_mode = "human"
    env = gym.make('manipulator_mujoco/DualUR5eEnv-v0', render_mode=render_mode)

    # demo = PegInHoleDemo("align", env, max_steps=500, max_demos=10, render_mode=render_mode)
    # demo = PegInHoleDemo("contact", env, max_steps=1000, max_demos=10, render_mode=render_mode)
    # demo = PegInHoleDemo("slide_insert", env, max_steps=2000, max_demos=10, render_mode=render_mode, seed=2024)
    demo = PegInHoleDemo("full", env, max_steps=2000, max_demos=10, render_mode=render_mode, seed=2024)
    demo.run()

    # recorder = DemoRecorder(env, record_rate=10)
    # scheduler = DemoScheduler(env)

    # # Ready position
    # scheduler.add_keyframe(
    #     left_pos = [-0.3, -0.6, 0.7, 0, -0.70710677, 0, 0.70710677],
    #     right_pos = [0.3, -0.6, 0.7, 0, 0.70710677, 0, 0.70710677],
    #     error_thresh= 1e-2,
    #     record=False
    # )

    # # Move in for contact
    # scheduler.add_keyframe(
    #     left_pos = [-0.15, -0.6, 0.7, 0, -0.70710677, 0, 0.70710677],
    #     right_pos = [0.15, -0.6, 0.7, 0, 0.70710677, 0, 0.70710677],
    #     error_thresh= 2e-2,
    #     wait_time=300,
    #     record=True
    # )

    # # Slide action
    # scheduler.add_keyframe(
    #     left_pos = [-0.15, -0.6, 0.7, 0, -0.70710677, 0, 0.70710677],
    #     right_pos = [0.15, -0.6, 0.7, 0, 0.70710677, 0, 0.70710677],
    #     error_thresh= 2e-2,
    #     wait_time=300,
    #     record=True
    # )

    # demo = Demo(env, scheduler, recorder, render_mode=render_mode)
    # demo.run()


if __name__ == '__main__':
    main()