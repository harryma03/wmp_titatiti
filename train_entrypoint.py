import numpy as np
import os
from datetime import datetime

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # High-obstacle + tilt finetune from the clean best checkpoint. Keep
    # WMP_titatit_best untouched so we can fall back if the high-level branch
    # trades away too much posture quality.
    train_cfg.runner.run_name = 'WMP_titatit_1_from_best'
    train_cfg.runner.load_run = 'WMP_titatit_best'
    train_cfg.runner.checkpoint = 14200
    train_cfg.runner.max_iterations = 20000
    train_cfg.runner.save_interval = 100
    train_cfg.runner.resume = True

    train_cfg.algorithm.learning_rate = 5.e-4
    train_cfg.algorithm.entropy_coef = 0.004
    train_cfg.algorithm.num_learning_epochs = 5

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_wmp_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    args.rl_device = args.sim_device
    train(args)
