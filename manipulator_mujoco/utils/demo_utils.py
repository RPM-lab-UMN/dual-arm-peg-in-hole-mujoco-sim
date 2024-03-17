import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Records and saves relevant values when running/recording scripted demonstrations
class DemoRecorder:
    def __init__(self, env, record_rate=10):
        self.env = env

        self.record_rate = record_rate
        self.time = 0
        self.times = []

        self.forces_left = []
        self.torques_left = []

        self.forces_right = []
        self.torques_right = []

        self.rgb_frames = {
            'overhead': [],
            'wrist_left': [],
            'wrist_right': []
        }

        self.video_frames = []

    def reset(self):
        self.time = 0
        self.times = []

        self.forces_left = []
        self.torques_left = []

        self.forces_right = []
        self.torques_right = []

        self.rgb_frames = {
            'overhead': [],
            'wrist_left': [],
            'wrist_right': []
        }

        self.video_frames = []

    def record_demo_values(self):
        self.times.append(self.time)

        self.forces_left.append(self.env.physics.bind(self.env.left_arm.force_sensor).sensordata.copy())
        self.torques_left.append(self.env.physics.bind(self.env.left_arm.torque_sensor).sensordata.copy())

        self.forces_right.append(self.env.physics.bind(self.env.right_arm.force_sensor).sensordata.copy())
        self.torques_right.append(self.env.physics.bind(self.env.right_arm.torque_sensor).sensordata.copy())

        cur_frame_overhead = self.env.render_frame(camera_id=0)[:,:,[2,1,0]]
        cur_frame_wrist_right = self.env.render_frame(camera_id=1)[:,:,[2,1,0]]
        cur_frame_wrist_left = self.env.render_frame(camera_id=2)[:,:,[2,1,0]]

        cur_f_plot = self.get_current_force_plot_left()
        cur_t_plot = self.get_current_torque_plot_left()

        self.rgb_frames['overhead'].append(cur_frame_overhead)
        self.rgb_frames['wrist_left'].append(cur_frame_wrist_left)
        self.rgb_frames['wrist_right'].append(cur_frame_wrist_right)

        top_row = np.hstack([cur_frame_overhead, cur_frame_wrist_left, cur_frame_wrist_right])
        bottom_row = np.hstack([cur_frame_overhead, cur_f_plot, cur_t_plot])

        self.video_frames.append(np.vstack([top_row, bottom_row]))

    def step(self):
        if self.time % self.record_rate == 0:
            self.record_demo_values()
        self.time += 1

    # Saves:
    #   RGB (overhead, wrist cameras)
    #   force-torque (left/right arms)
    #   proprioception
    #   mp4 video
    def save_recording(self):
        size = self.video_frames[0].shape
        print("Saving video...")
        out = cv2.VideoWriter('yuh.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (1920,960)) # (1920,480)
        for i in range(len(self.video_frames)):
            out.write(self.video_frames[i])
        out.release()
        print("Done!")

    def get_current_force_plot_left(self):
        fig = plt.figure()
        plt.title("Left Arm Forces")
        plt.ylim(-200, 200)
        plt.plot(self.times[-40:], [x[0] for x in self.forces_left][-40:], linestyle="-", marker=".", markersize=1, color="r", label="force-x")
        plt.plot(self.times[-40:], [x[1] for x in self.forces_left][-40:], linestyle="-", marker=".", markersize=1, color="g", label="force-y")
        plt.plot(self.times[-40:], [x[2] for x in self.forces_left][-40:], linestyle="-", marker=".", markersize=1, color="b", label="force-z")
        plt.legend(loc="lower right")
        
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()

        return data

    def get_current_torque_plot_left(self):
        fig = plt.figure()
        plt.title("Left Arm Torques")
        plt.ylim(-3, 3)
        plt.plot(self.times[-40:], [x[0] for x in self.torques_left][-40:], linestyle="-", marker=".", markersize=1, color="r", label="torque-x")
        plt.plot(self.times[-40:], [x[1] for x in self.torques_left][-40:], linestyle="-", marker=".", markersize=1, color="g", label="torque-y")
        plt.plot(self.times[-40:], [x[2] for x in self.torques_left][-40:], linestyle="-", marker=".", markersize=1, color="b", label="torque-z")
        plt.legend(loc="lower right")
        
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()

        return data
    
    def get_current_force_plot_right(self):
        fig = plt.figure()
        plt.title("Right Arm Forces")
        plt.ylim(-200, 200)
        plt.plot(self.times[-40:], [x[0] for x in self.forces_right][-40:], linestyle="-", marker=".", markersize=1, color="r", label="force-x")
        plt.plot(self.times[-40:], [x[1] for x in self.forces_right][-40:], linestyle="-", marker=".", markersize=1, color="g", label="force-y")
        plt.plot(self.times[-40:], [x[2] for x in self.forces_right][-40:], linestyle="-", marker=".", markersize=1, color="b", label="force-z")
        plt.legend(loc="lower right")
        
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()

        return data

    def get_current_torque_plot_right(self):
        fig = plt.figure()
        plt.title("Right Arm Torques")
        plt.ylim(-3, 3)
        plt.plot(self.times[-40:], [x[0] for x in self.torques_right][-40:], linestyle="-", marker=".", markersize=1, color="r", label="torque-x")
        plt.plot(self.times[-40:], [x[1] for x in self.torques_right][-40:], linestyle="-", marker=".", markersize=1, color="g", label="torque-y")
        plt.plot(self.times[-40:], [x[2] for x in self.torques_right][-40:], linestyle="-", marker=".", markersize=1, color="b", label="torque-z")
        plt.legend(loc="lower right")
        
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()

        return data

class DemoScheduler:
    def __init__(self, env, verbose=False):
        self.env = env
        self.phase = 0
        self.wait_time = 0
        self.verbose = verbose

        self.keyframes = []

    def reset(self):
        self.phase = 0
        self.wait_time = 0

    def add_keyframe(self, left_pos, right_pos, error_thresh=1e-2, wait_time=0, record=True):
        self.keyframes.append({
            'left_pos': left_pos,
            'right_pos': right_pos,
            'error_thresh': error_thresh,
            'wait_time': wait_time,
            'record': record
        })
    
    def step(self):
        if self.is_complete():
            # Demo is complete
            return self.keyframes[-1]['left_pos'], self.keyframes[-1]['right_pos']

        left_eef_pose = self.env.left_arm.get_eef_pose(self.env.physics)
        right_eef_pose = self.env.right_arm.get_eef_pose(self.env.physics)

        left_target_pose = self.keyframes[self.phase]['left_pos']
        right_target_pose = self.keyframes[self.phase]['right_pos']

        error_thresh = self.keyframes[self.phase]['error_thresh']
        wait_time = self.keyframes[self.phase]['wait_time']

        left_pose_error = np.linalg.norm(left_eef_pose - left_target_pose) / np.linalg.norm(left_target_pose)
        right_pose_error = np.linalg.norm(right_eef_pose - right_target_pose) / np.linalg.norm(right_target_pose)

        if self.verbose:
            print('='*10)
            print(left_pose_error)
            print(right_pose_error)
            print('='*10)

        self.wait_time += 1
        
        if self.wait_time > wait_time and left_pose_error < error_thresh and right_pose_error < error_thresh:
            self.wait_time = 0
            self.phase += 1

            if self.is_complete():
                # Demo is complete
                return self.keyframes[-1]['left_pos'], self.keyframes[-1]['right_pos']

        new_left_target_pose = self.keyframes[self.phase]['left_pos']
        new_right_target_pose = self.keyframes[self.phase]['right_pos']
            
        # Return current action
        return left_target_pose, right_target_pose
    
    def can_record(self):
        if not self.is_complete():
            return self.keyframes[self.phase]['record']
        else:
            return False
    
    def is_complete(self): 
        return self.phase >= len(self.keyframes)


class Demo:
    def __init__(
            self,
            env: gym.Env,
            demo_scheduler: DemoScheduler,
            demo_recorder: DemoRecorder,
            max_steps=-1,
            render_mode=None
    ):
        self.env = env
        self.scheduler = demo_scheduler
        self.recorder = demo_recorder
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode

    def reset(self):
        self.env.reset()
        self.recorder.reset()
        self.scheduler.reset()
    
    def run(self):
        self.reset()

        if self.render_mode == "human":
            while True:
                left_action, right_action = self.scheduler.step()
                self.env.step(np.array([left_action, right_action]))
                self.env.render_frame(camera_id=0)

                if self.scheduler.is_complete():
                    self.reset()
        else:
            while not self.scheduler.is_complete():
                left_action, right_action = self.scheduler.step()
                self.env.step(np.array([left_action, right_action]))
                if self.scheduler.can_record():
                    self.recorder.step()
            print("Demo complete: Saving data...")
            self.recorder.save_recording()
