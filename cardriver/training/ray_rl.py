# %%

from cardriver.env import SimpleRoad
from collections import namedtuple
from datetime import datetime
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from time import sleep
import gym
import gym.core
import numpy as np
import pandas as pd
import ray, gc
import ray.rllib.utils
import shutil
import unittest.mock

from cardriver.utils import show_hard_test_run, show_test_run

gym.core.RenderFrame = unittest.mock.Mock()

# %%
ray.shutdown()
sleep(1)
ray.init(ignore_reinit_error=True)

algo = None
gc.collect()
config = (
    PPOConfig()
    .environment(SimpleRoad, env_config={'render_mode': 'ascii'}, render_env=False)
    # .rollouts(num_rollout_workers=4, num_envs_per_worker=50)
    .framework("torch")#, eager_tracing=True)
    .resources(num_cpus_per_worker=5)
    # .training(model={"fcnet_hiddens": [64, 64]})
    .evaluation(evaluation_num_workers=1)
)

algo = config.build()
epochs = 0

# %%
for _ in range(20):
    epochs += 1
    try:
        full_info = algo.train()
        info = {
            'epochs': epochs,
            'date': datetime.now().isoformat(),
            'total_loss': full_info['info']['learner']['default_policy']['learner_stats']['total_loss'],
            'episode_reward_max': full_info['sampler_results']['episode_reward_max'],
            'episode_reward_min': full_info['sampler_results']['episode_reward_min'],
            'episode_reward_mean': full_info['sampler_results']['episode_reward_mean'],
            'sampler_perf': full_info['sampler_results']['sampler_perf'],
            'num_agent_steps_trained': full_info['num_agent_steps_trained'],
            'num_env_steps_sampled_this_iter': full_info['num_env_steps_sampled_this_iter'],
        }
        print(info)
    except Exception as e: sleep(1); print('ERROR', e)
    if epochs % 20 == 0:
        print(full_info)
        shutil.rmtree('/tmp/raycardriver/', ignore_errors=True)
        algo.save('/tmp/raycardriver/')
        print('Saved')
# %%
algo.evaluate()
# %%
algo.compute_single_action(([10., 8.3, 10.], 0))

# %%
def model(observation, state):
    return algo.compute_single_action(observation, full_fetch=True, state=state)
show_test_run(model)

#%%
show_hard_test_run(model)
