import os

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"
import pyrender

from calvin_env.envs.play_table_env import get_env

# NOTE: Please download the CALVIN dataset and put it in the path
path = "/mnt/bn/robotics/manipulation_data/calvin_data/calvin_debug_dataset/validation"
env = get_env(path, show_gui=False)
print(env.get_obs())
