import actpermoma

import os
from copy import deepcopy as cp
import numpy as np
from itertools import product

from experiment_launcher import Launcher

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from actpermoma.utils.hydra_cfg.hydra_utils import *


LOCAL = True#is_local()
TEST = False
USE_CUDA = True

N_SEEDS = 1

N_CORES = 4

MEMORY_PER_CORE = 13000# 13000
PARTITION = 'gpu' # 'gpu -C [rtx3090]' # 'rtx2', 'rtx'
GRES = 'gpu' # if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = 'looking_good'

exp_name = 'active_grasp'

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs_and_launch_exp(cfg: DictConfig):

    import datetime
    time = datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')

    base_dir = os.path.join(actpermoma.__path__[0],'logs', time)

    launcher = Launcher(
        exp_name='eval_'+exp_name,
        exp_file='active_grasp_pipeline', # local path without .py    
        # project_name='project01907',  # for hrz cluster
        base_dir=base_dir,
        n_seeds=N_SEEDS,
        n_cores=N_CORES,
        memory_per_core=MEMORY_PER_CORE,
        days=0,
        hours=4,
        minutes=59,
        seconds=0,
        partition=PARTITION,
        gres=GRES,
        conda_env=CONDA_ENV,
        use_timestamp=True,
        compact_dirs=False
    )

    ## parameters
    parameters = {
                  'quality_thresholds': [0.6, 0.8],
                #   'gain_weighting_factors': [0.2, 0.5, 0.8, 3.0],
                #   'seen_theta_threshs': [0, 30*np.pi/180, 45*np.pi/180],
                #   'momentums': [0, 700, 800, 1000],
                # #   'base_goals_radius_grasps': [0.75, 0.8],
                  'window_sizes': [1, 5],
                  }

    defaults = {
                'quality_threshold': 0.8,
                # 'gain_weighting_factor': 0.5,
                # 'seen_theta_thresh': 45*np.pi/180,
                # 'momentum': 800,
                # # 'base_goals_radius_grasp': 0.75,
                'window_size': 1,
                }

    param_combinations = []
    for key in parameters.keys():
        for key_val in parameters[key]:
            param_combi = cp(defaults)
            param_combi[key[:-1]] = key_val
            param_combi['spec'] = key[:-1]

            param_combinations.append(param_combi)
    
    for param_combi in param_combinations:
        # src: https://stackoverflow.com/questions/66295334/create-a-new-key-in-hydra-dictconfig-from-python-file
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.name_spec = 'scene_type='+str(cfg.task.env.scene_type) + '_basic_'
            cfg.short_name_spec = cfg.name_spec
            special_key = param_combi['spec']
            del param_combi['spec']
            for key in param_combi.keys():
                if key is 'seen_theta_thresh':
                    val = str(np.rad2deg(param_combi[key]))
                else:
                    val = str(param_combi[key])
                
                cfg.name_spec += '_' + key + '=' + val

                if key == special_key:
                    cfg.short_name_spec += '_' + key + '=' + val
            
        agent_params_cfg = cfg.train.params.config
        for key in param_combi.keys():
            agent_params_cfg[key] = param_combi[key]               

        results_dir = os.path.join(base_dir, 'cfgs', str(cfg.name_spec))

        # Save experiment config in a file at the right path
        try:
            os.makedirs(results_dir)
        except OSError:
            # directory already exists implying that this experiment is already launched
            continue
        exp_cfg_file_path = os.path.join(results_dir, 'config.yaml')
        with open(exp_cfg_file_path, 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))
        
        launcher.add_experiment(cfg_file_path__=exp_cfg_file_path)
        
    launcher.run(LOCAL, TEST)


if __name__ == '__main__':

    parse_hydra_configs_and_launch_exp()