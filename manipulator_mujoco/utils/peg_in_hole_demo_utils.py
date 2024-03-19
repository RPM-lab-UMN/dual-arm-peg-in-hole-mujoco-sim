import numpy as np

from manipulator_mujoco.utils import Demo, DemoRecorder, DemoScheduler

class PegInHoleDemo(Demo):
    def __init__(self, phase, env, render_mode, max_demos=1, max_steps=-1, seed=0):
        assert phase in ["align", "contact", "slide_insert", "retreat"]
        self.phase = phase
        self.trans_rng = np.random.default_rng(seed)

        demo_recorder = DemoRecorder(env, record_rate=10)
        demo_scheduler = self.build_phase_scheduler(DemoScheduler(env, verbose=False), phase)

        super(self.__class__, self).__init__(env, demo_scheduler, demo_recorder, max_demos=max_demos, max_steps=max_steps, render_mode=render_mode)
    
    def build_phase_scheduler(self, scheduler, phase):
        if phase == "align":
            base_align_pose_left = [-0.3, -0.6, 0.7, 0, -0.70710677, 0, 0.70710677]
            base_align_pose_right = [0.3, -0.6, 0.7, 0, 0.70710677, 0, 0.70710677]

            diff_align_pose_left = base_align_pose_left.copy()
            diff_align_pose_right = base_align_pose_right.copy()

            perturb_left = self.generate_trans_perturbation(min_val=-0.05, max_val=0.05)
            perturb_right = self.generate_trans_perturbation(min_val=-0.05, max_val=0.05)

            diff_align_pose_left[:3] += perturb_left
            diff_align_pose_right[:3] += perturb_right

            base_align_pose_left[0] = diff_align_pose_left[0]
            base_align_pose_right[0] = diff_align_pose_right[0]

            # Moving to perturbed position
            scheduler.add_keyframe(
                left_pos = diff_align_pose_left,
                right_pos = diff_align_pose_right,
                error_thresh = 1e-2,
                record = False
            )

            # Moving to aligned position
            scheduler.add_keyframe(
                left_pos = base_align_pose_left,
                right_pos = base_align_pose_right,
                error_thresh = 1e-2,
                record = True
            )

        elif phase == "contact":
            base_contact_pose_left = [-0.3, -0.6, 0.7, 0, -0.70710677, 0, 0.70710677]
            base_contact_pose_right = [0.3, -0.6, 0.7, 0, 0.70710677, 0, 0.70710677]

            perturb_left = self.generate_trans_perturbation(min_val=-0.025, max_val=0.025)
            perturb_right = self.generate_trans_perturbation(min_val=-0.025, max_val=0.025)

            base_contact_pose_left[:3] += perturb_left
            base_contact_pose_right[:3] += perturb_right

            final_contact_pose_left = base_contact_pose_left.copy()
            final_contact_pose_right = base_contact_pose_right.copy()

            final_contact_pose_left[0] = -0.15
            final_contact_pose_right[0] = 0.15

            # Moving to perturbed position
            scheduler.add_keyframe(
                left_pos = base_contact_pose_left,
                right_pos = base_contact_pose_right,
                error_thresh = 1e-2,
                record = False
            )

            # Moving to aligned position
            scheduler.add_keyframe(
                left_pos = final_contact_pose_left,
                right_pos = final_contact_pose_right,
                error_thresh = 2e-2,
                wait_time = 300,
                record = True
            )            
        
        return scheduler
    
    def reset(self):
        self.scheduler = self.build_phase_scheduler(DemoScheduler(self.env, verbose=False), self.phase)
        super().reset()

    def generate_trans_perturbation(self, min_val, max_val):
        delta = self.trans_rng.uniform(-max_val,max_val,(3,))
        delta[0] *= 0.5 # Scale z perturbation to not be as large
        return delta


