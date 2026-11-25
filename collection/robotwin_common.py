"""Local real RoboTwin adapter. No physics, planner, camera or success mocks.

RoboTwin main + official RLinf_support mplib robot/planner; clean procedural
stack_blocks_two task, raster renderer. See 验证记录.md for deviations.
"""
import os,sys,json,time
from pathlib import Path
import numpy as np
from dataset import SMOL_ROOT, ROBOTWIN_ROOT, add_paths
add_paths('sim')
N=SMOL_ROOT/'.runtime/robotwin';T=ROBOTWIN_ROOT
N.mkdir(parents=True, exist_ok=True)
# Vulkan ICD selection can be configured by the caller.
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.chdir(T);sys.path.insert(0,str(T))
import yaml
from envs.stack_blocks_two import stack_blocks_two

def make_env(seed=0,collect=False,embodiment='aloha-agilex'):
    args=yaml.safe_load((T/'env_cfg/task_config/demo_clean.yml').read_text())
    robot_dir=T/'assets/embodiments'/embodiment
    robot_cfg=yaml.safe_load((robot_dir/'config.yml').read_text())
    for cam in robot_cfg['static_camera_list']:cam.setdefault('type','D435')
    args.update(task_name='stack_blocks_two',task_config='audit_clean',
        seed=seed,ep_num=0,render_freq=0,save_freq=30,collect_data=collect,
        save_path=str(N/'collected'),eval_mode=not collect,planner_backend='mplib',
        left_robot_file=str(robot_dir),right_robot_file=str(robot_dir),
        left_embodiment_config=robot_cfg,right_embodiment_config=robot_cfg,
        dual_arm_embodied=embodiment=='aloha-agilex',embodiment_dis=.6,embodiment_name=embodiment)
    env=stack_blocks_two();env.setup_demo(**args)
    return env,args

def observe(env):
    obs=env.get_obs()
    return obs['observation']['head_camera']['rgb'].copy(),np.asarray(obs['joint_action']['vector'],dtype=np.float32)

def valid_action(env,a):
    """Absolute joint targets in radians; grippers at indices 6/13 are [0,1]."""
    a=np.asarray(a,dtype=np.float64).reshape(14).copy()
    assert np.isfinite(a).all()
    for arm,off in [('left',0),('right',7)]:
        for j,joint in enumerate(getattr(env.robot,arm+'_arm_joints')):
            bounds=joint.get_limits()[0];a[off+j]=np.clip(a[off+j],bounds[0],bounds[1])
    a[[6,13]]=np.clip(a[[6,13]],0,1)
    return a

if __name__=='__main__':
    from PIL import Image
    e,c=make_env();im,q=observe(e);Image.fromarray(im).save(N/'robotwin_initial.png')
    print('ENV_OK',im.shape,q,flush=True)
    e.take_action(q,action_type='qpos');im,q2=observe(e)
    (N/'environment.json').write_text(json.dumps({'image_shape':list(im.shape),'state':q.tolist(),'state_after_hold':q2.tolist(),'physical_step':True,'renderer':'raster/default','planner':'official RLinf_support MplibWrapperPlanner','task':'stack_blocks_two'},indent=2))
    e.close_env()
