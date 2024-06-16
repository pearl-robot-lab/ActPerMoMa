import os
from datetime import datetime
import argparse
import itertools
from copy import deepcopy as cp

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import actpermoma
from actpermoma.utils.hydra_cfg.hydra_utils import *
from actpermoma.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from actpermoma.utils.task_util import initialize_task
from actpermoma.envs.isaac_env_mushroom import IsaacEnvMushroom
from actpermoma.algos.active_grasp import ActiveGraspAgent
from actpermoma.algos.naive import NaiveAgent
from actpermoma.algos.random import RandomAgent
from actpermoma.algos.actpermoma import ActPerMoMaAgent
# Use Mushroom RL library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.actor_critic import *
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from tqdm import trange

def parse_data_and_save(eval_dataset, results_dir, name_spc=None, wandb_logger=None):
    # eval_dataset is a list of tuples respresenting (s, a, r, s', absorbing, info, last); where s is a dict with keys 'tsdf', 'goal_pos', 'view_pos', 'view_theta', 'view_phi', 'obj_states', 'reset_agent'
    # Save dataset
    distances_list = []
    num_views_list = []
    successes_list = []
    failures_list = []
    aborted_list = []
    num_IK_fails_list = []
    num_grasps_attempted_list = []
    theta_covered_list = []

    # np.save(os.path.join(results_dir, 'eval_dataset.npy'), eval_dataset)
    # Save samples
    for i, (s, a, r, s_, absorbing, info, last) in enumerate(eval_dataset):
        if 'tsdf' in s: del s['tsdf']
        if 'tsdf' in s_: del s_['tsdf']
        if last:
            distances_list.append(info['distance'][0])
            num_views_list.append(info['num_views'][0])
            successes_list.append(info['grasp_success'][0])
            failures_list.append(info['grasp_failure'][0])
            aborted_list.append(info['aborted'][0])
            num_IK_fails_list.append(info['num_IK_fails'][0])
            num_grasps_attempted_list.append(info['num_grasps_attempted'][0])
            theta_covered_list.append(info['theta_covered'][0])

    d_mean = np.mean(distances_list)
    d_std = np.std(distances_list)
    nv_mean = np.mean(num_views_list)
    nv_std = np.std(num_views_list)
    nIK_mean = np.mean(num_IK_fails_list)
    nIK_std = np.std(num_IK_fails_list)
    ngrasps_mean = np.mean(num_grasps_attempted_list)
    ngrasps_std = np.std(num_grasps_attempted_list)
    th_mean = np.mean(theta_covered_list)
    th_std = np.std(theta_covered_list)
    sr = np.sum(successes_list) / len(successes_list)
    fr = np.sum(failures_list) / len(failures_list)
    ar = np.sum(aborted_list) / len(aborted_list)

    metrics = {'distance_mean': d_mean, 
               'distance_std': d_std, 
               'num_views_mean': nv_mean, 
               'num_views_std': nv_std, 
               'num_IK_fails_mean': nIK_mean,
               'num_IK_fails_std': nIK_std,
               'num_grasps_att_mean': ngrasps_mean,
               'num_grasps_att_std': ngrasps_std,
               'theta_covered_mean': th_mean,
               'theta_covered_std': th_std,
               'success_rate': sr, 
               'failure_rate': fr, 
               'aborted_rate': ar}

    print(metrics)

    if name_spc is None:
        name_spc = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if wandb_logger is not None:
        wandb_logger.log_metrics(metrics, name_spc)
        wandb_logger.finish()
    np.save(os.path.join(results_dir, f'eval_dataset-{name_spc}.npy'), eval_dataset, allow_pickle=True)
    np.save(os.path.join(results_dir, f'metrics-{name_spc}.npy'), metrics, allow_pickle=True)

    return d_mean, nv_mean, sr, fr, ar

def experiment(cfg: DictConfig = None, cfg_file_path: str = "", seed: int = 0, results_dir: str = "", env=None, task=None):
    # Get configs
    if (cfg_file_path):
        # Get config file from path
        cfg = OmegaConf.load(cfg_file_path)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)
    headless = cfg.headless
    render = cfg.render
    sim_app_cfg_path = cfg.sim_app_cfg_path
    agent_params_cfg = cfg.train.params.config
    algo_map = {"ActiveGrasp": ActiveGraspAgent, 
                "ActPerMoMa": ActPerMoMaAgent,
                "Naive": NaiveAgent,
                "Random": RandomAgent}
    algo = algo_map[cfg.train.params.algo.name]
    algo_name = cfg.train.params.algo.name

    # Set up environment
    if env is None or task is None:
        env = IsaacEnvMushroom(headless=headless, render=render, sim_app_cfg_path=sim_app_cfg_path)
        task = initialize_task(cfg_dict, env)
    render=False # TODO

    # Set up logging paths/directories
    exp_name = cfg.train.params.config.name
    exp_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  # append datetime for logging
    results_dir = os.path.join(actpermoma.__path__[0], 'logs', cfg.task.name, exp_name)
    if (cfg.test): results_dir = os.path.join(results_dir, 'test')
    results_dir = os.path.join(results_dir, exp_stamp)
    os.makedirs(results_dir, exist_ok=True)
    # log experiment config
    with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    obs_to_states = [task.obs_to_state]


    # Loop over num_seq_seeds:
    for exp in range(cfg.num_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        seed += 1

        # Logger
        logger = Logger(results_dir=results_dir, log_console=True)
        logger.strong_line()
        # if cfg.name_spec exists
        if hasattr(cfg, 'name_spec'):
            name_spec = cfg.name_spec
        else:
            name_spec = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        logger.info(f'Experiment: {exp_name}, Algo{algo_name}: ' + name_spec)

        exp_eval_dataset = list()  # This will be a list of dicts with datasets from every epoch
        if cfg.wandb_activate is True:
            # (Optional) Logging with Weights & biases
            from actpermoma.utils.wandb_utils import wandbLogger
            wandb_logger = wandbLogger(exp_config=cfg, run_name=logger._log_id, group_name=algo_name+'_'+exp_stamp) # Optional
            if hasattr(cfg, 'short_name_spec'):
                wandb_logger = wandbLogger(exp_config=cfg, group_name=algo_name+'_'+cfg.short_name_spec+'_'+exp_stamp)
            else:
                wandb_logger = wandbLogger(exp_config=cfg, group_name=algo_name+'_'+exp_stamp) # Optional
        else:
            wandb_logger = None

        # Agent
        if not cfg_dict["task"] is None and cfg_dict["task"]["name"] == 'TiagoDualActivePerception' or cfg_dict["task"]["name"] == 'TiagoDualActivePerceptionNaive':
            # src: https://stackoverflow.com/questions/66295334/create-a-new-key-in-hydra-dictconfig-from-python-file
            OmegaConf.set_struct(agent_params_cfg, True)
            with open_dict(agent_params_cfg):
                agent_params_cfg['fx'] = cfg_dict["task"]["env"]["fx"]
                agent_params_cfg['fy'] = cfg_dict["task"]["env"]["fy"]
                agent_params_cfg['cx'] = cfg_dict["task"]["env"]["cx"]
                agent_params_cfg['cy'] = cfg_dict["task"]["env"]["cy"]
        else:
            raise Exception("No task config provided, but camera parameters required")
        
        agent = algo(env.info, agent_params_cfg)
        agent.tiago_handler = task.tiago_handler # didn't figure out how to pass this to the agent in a cleaner way

        # Algorithm
        core = Core(agent, env, preprocessors=obs_to_states)

        # RUN
        eval_dataset = core.evaluate(n_episodes=agent_params_cfg.n_episodes_test, render=render)
        # eval_dataset is a list of tuples respresenting (s, a, r, s', absorbing, info, last); where s is a dict with keys 'tsdf', 'goal_pos', 'view_pos', 'view_theta', 'view_phi', 'obj_states', 'reset_agent'
        d_mean, nv_mean, sr, fr, ar = parse_data_and_save(eval_dataset, results_dir, name_spc=f'Algo{algo_name}_' + name_spec, wandb_logger=wandb_logger)

        logger.epoch_info(0, success_rate=sr, avg_episode_length=nv_mean)
        # Optional wandb logging
        # wandb_logger.run_log_wandb(success_rate, J, R, E, avg_episode_length, q_loss)

        # Get video snippet of final learnt behavior (enable internal rendering for this)
        # prev_env_render_setting = bool(env._run_sim_rendering)
        # env._run_sim_rendering = True
        # img_dataset = core.evaluate(n_episodes=5, get_renders=True)
        # env._run_sim_rendering = prev_env_render_setting
        # log dataset and video
        # logger.log_dataset(exp_eval_dataset)
        # run_log_wandb(exp_config=cfg, run_name=logger._log_id, group_name=exp_name+'_'+exp_stamp, dataset=exp_eval_dataset)
        # img_dataset = img_dataset[::15] # Reduce size of img_dataset. Take every 15th image
        # wandb_logger.vid_log_wandb(img_dataset=img_dataset)

    # Shutdown
    env._simulation_app.close()


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs_and_run_exp(cfg: DictConfig):
    experiment(cfg)


if __name__ == '__main__':
    #parse_hydra_configs_and_run_exp()
    #parser = argparse.ArgumentParser()
    #parser.add_argument('results_dir')
    #parser.add_argument('seed')
    #parser.add_argument('cfg_file_path')
    #args = parser.parse_args()
    #experiment(cfg_file_path = args.cfg_file_path, seed=args.seed, results_dir=args.results_dir)
    parser = argparse.ArgumentParser("Train/Test agent: (Local or SLURM)")
    parser.add_argument("--cfg_file_path", type=str, default="", help="Optional config file to run experiment (typically when using slurm)")
    args, _ = parser.parse_known_args()
    
    parse_hydra_configs_and_run_exp()