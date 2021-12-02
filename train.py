from src.dqn_agent import DqnAgent
from src.lib.preprocessing import make_env
import gym
import json


if __name__=="__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    config = config['config_vanilla_without_target_rescale']

    env_name = 'SeaquestNoFrameskip-v4'

    env = make_env(env_name, shape=(42, 42, 1), repeat=4, 
                  clip_rewards=False, no_ops=0, fire_first=False)



    dqn = DqnAgent(env, env_name, config)
    dqn.train_agent()
