#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
# import mujoco_py  # Mujoco must come before other imports. https://openai.slack.com/archives/C1H6P3R7B/p1492828680631850
from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import gym
import logging
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
import trpo_mpi
from rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize
import sys
from tester import Tester
from util import TimeStepHolder
from SLBDAO.common import models, variables, custom_session
import tensorflow as tf


def train(env_id, num_timesteps, seed):
    import baselines.common.tf_util as U
    # sess = U.single_threaded_session()
    # sess.__enter__()
    gpu_options = tf.GPUOptions(
        allow_growth=False, per_process_gpu_memory_fraction=0.2)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=tf_config)
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    # env = gym.make(env_id)
    env = normalize(InvertedDoublePendulumEnv(), normalize_obs=False)
    env_t = normalize(InvertedDoublePendulumEnv(), normalize_obs=False)

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
                         hid_size=32, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), str(rank)),  allow_early_resets=True)
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)
    time_step_holder = TimeStepHolder(0, 0)
    tester = Tester(episodes=100, period=10, env=env_t, time_step_holder=time_step_holder, file='./results', session=sess)
    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                   max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3, tester=tester)
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
