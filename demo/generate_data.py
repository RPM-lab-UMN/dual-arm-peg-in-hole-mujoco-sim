import manipulator_mujoco
import gymnasium as gym

from manipulator_mujoco.utils import Demo, DemoRecorder, DemoScheduler

def main():
    env = gym.make('manipulator_mujoco/DualUR5eEnv-v0', render_mode="human")

    recorder = DemoRecorder(env, record_rate=10)
    scheduler = DemoScheduler(env)

    # Ready position
    scheduler.add_keyframe(
        left_pos = [-0.3, -0.6, 0.71, 0, -0.70710677, 0, 0.70710677],
        right_pos = [0.3, -0.6, 0.69, 0, 0.70710677, 0, 0.70710677],
        error_thresh= 1e-2
    )

    # Move in for contact
    scheduler.add_keyframe(
        left_pos = [-0.15, -0.6, 0.71, 0, -0.70710677, 0, 0.70710677],
        right_pos = [0.15, -0.6, 0.69, 0, 0.70710677, 0, 0.70710677],
        error_thresh= 2e-2,
        wait_time=275
    )

    # Slide action
    scheduler.add_keyframe(
        left_pos = [-0.15, -0.6, 0.7, 0, -0.70710677, 0, 0.70710677],
        right_pos = [0.15, -0.6, 0.7, 0, 0.70710677, 0, 0.70710677],
        error_thresh= 2e-2,
        wait_time=300
    )

    demo = Demo(env, scheduler, recorder, render_mode="human")
    demo.run()


if __name__ == '__main__':
    main()