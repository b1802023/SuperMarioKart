import re
import retro
import os

from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib.ppo_recurrent import RecurrentPPO

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from wrappers import SuperMarioKartDiscretizer, CustomRewarpAndDoneEnv, StochasticFrameSkip, TimeLimit, Processing_wapper, TensorboardCallback, Render_env
from model import CustomCNN

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

def make_env(log_dir):
    env = retro.make(game='SuperMarioKart', state='MarioCircuit1_time')
    # env = retro.make(game='SuperMarioKart', state='KinokoCap50')
    env.seed(0)

    env = SuperMarioKartDiscretizer(env)
    #env = Render_env(env) # ゲーム画面のレンダー用wrapper
    env = CustomRewarpAndDoneEnv(env)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = TimeLimit(env, max_episode_steps=2700)
    env = Processing_wapper(env, 2)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env)
    
    return env

def main():
    log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)
    
    env = make_env(log_dir)
    
    policy_kwargs = dict(
    features_extractor_class = CustomCNN
    )

    model = RecurrentPPO('MultiInputLstmPolicy', 
                        env, 
                        n_steps=2048,
                        batch_size=64,
                        policy_kwargs=policy_kwargs,
                        verbose=1, 
                        tensorboard_log=log_dir)

    callback = TensorboardCallback(check_freq = 1000, log_dir = log_dir)
    model.learn(total_timesteps=2000000, callback = callback)
    model.save('PPO_model')

if __name__ == "__main__":
    main()