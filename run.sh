MUJOCO_GL=egl python demo/ur5e_demo.py
MUJOCO_GL=egl python demo/generate_data.py

# Annoying github stuff
git stash push --include-untracked
git stash drop
git pull