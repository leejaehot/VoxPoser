import os
# os.environ["LD_LIBRARY_PATH"] = "/home/jclee/CoppeliaSim/libcoppeliaSim.so.1"

import openai # 0.28.1
from arguments import get_config # config 파일 불러와서 활용할 용도.
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from envs.rlbench_env import VoxPoserRLBench
from utils import set_lmp_objects
import numpy as np
from rlbench import tasks

import os
from dotenv import load_dotenv

load_dotenv()
# .env 에 기재된 OPENAI_API_KEY가 os.environ에 환경변수로 저장.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)


openai.api_key = OPENAI_API_KEY

config = get_config('rlbench')
# uncomment this if you'd like to change the language model (e.g., for faster speed or lower cost)
# for lmp_name, cfg in config['lmp_config']['lmps'].items():
#     cfg['model'] = 'gpt-3.5-turbo'

# initialize env and voxposer ui
visualizer = ValueMapVisualizer(config['visualizer'])

env = VoxPoserRLBench(visualizer=visualizer)
lmps, lmp_env = setup_LMP(env, config, debug=False)
voxposer_ui = lmps['plan_ui']



# below are the tasks that have object names added to the "task_object_names.json" file
# uncomment one to use
env.load_task(tasks.PutRubbishInBin)
#env.load_task(tasks.LampOff)
#env.load_task(tasks.OpenWineBottle)
# env.load_task(tasks.PushButton)
# env.load_task(tasks.TakeOffWeighingScales)
# env.load_task(tasks.MeatOffGrill)
# env.load_task(tasks.SlideBlockToTarget)
# env.load_task(tasks.TakeLidOffSaucepan)
# env.load_task(tasks.TakeUmbrellaOutOfUmbrellaStand)

descriptions, obs = env.reset()
set_lmp_objects(lmps, env.get_object_names())  # set the object names to be used by voxposer

instruction = np.random.choice(descriptions) # 명령어/









####
#lmp_env.detect("bottle")['occupancy_map'] # (100,100,100)
#lmp_env.detect("bottle")['name'] # 'bottle'
#lmp_env.detect("bottle")['aabb'] # [[49, 66, 0], [56, 72, 31]]
#lmp_env.detect("bottle")['_position_world'] # array([0.2838978 , 0.25511591, 0.92656969])
#lmp_env.detect("bottle")['_point_cloud_world'] # (1083,3) # downsampling된거.
#lmp_env.detect("bottle")['normal'] # array([-0.04232502, -0.77748427,  0.62747654])

#env.get_scene_3d_obs()
#getattr(env.latest_obs,f"{env.camera_names[4]}_rgb")
#getattr(env.latest_obs,f"{env.camera_names[4]}_depth")
#getattr(env.latest_obs,f"{env.camera_names[4]}_point_cloud")

import pdb;pdb.set_trace()
voxposer_ui(instruction)


env.rlbench_env.shutdown()