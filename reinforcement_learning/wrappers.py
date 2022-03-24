import cv2
import numpy as np
import gym
from gym import spaces
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

class CustomRewarpAndDoneEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewarpAndDoneEnv, self).__init__(env)
        self.check_point = 0
        self.lap_time = 0
        self.lap = 128
        
    def reset(self, **kwargs):
        self.check_point = 0
        self.lap_time = 0
        self.lap = 128
        
        return self.env.reset(**kwargs)

    def step(self, action):
        
        state, reward, done, info = self.env.step(action)
        speed_reward, rank_reward, check_point_reward, reward, lap_reward = 0, 0, 0, 0, 0
        
        # if  info['speed'] >= 700:
        #     speed_reward = 1
            
        if info['check_point'] > self.check_point:
            check_point_reward += 1
        elif info['check_point'] < self.check_point:
            check_point_reward -= 1
        else:
            check_point_reward = 0
            
        # if info['rank'] <= 4:
        #     rank_reward = 1
        
        reward = check_point_reward
        
        # if info['surface'] == 64:
        #     reward = reward
        # else:
        #     reward = reward * 0.6
        
        self.check_point = info['check_point']
        
        if info['lap'] > self.lap:
            lap_reward = (60 - ((self.c16to10(info['time_min']) * 60 + self.c16to10(info['time_sec'])) - self.lap_time)) / 5 
            reward += lap_reward if lap_reward > 0 else 0
            self.lap = info['lap']
            self.lap_time =  (self.c16to10(info['time_min']) * 60 + self.c16to10(info['time_sec']))
        
        if info['lap'] == 133:
            reward += 10
            done = True
            print('min:{}, sec:{}, milisec:{}'.format(self.c16to10(info['time_min']), self.c16to10(info['time_sec']), self.c16to10(info['time_milisec'])))
        
        
        return state, reward, done, info
    
    def c16to10(self, num):
        return (num // 16)*10 + num % 16
    
class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        
        self.count = 0 
        
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
            
        if done:
            self.count += 1
            
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
    
class Render_env(gym.Wrapper):
    def __init__(self, env):
        super(Render_env, self).__init__(env)

    def step(self, action):
        self.env.render()
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0
        for i in range(self.n):
            if self.curac is None:
                self.curac = ac
            elif i==0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            elif i==1:
                self.curac = ac
            if self.supports_want_render and i<self.n-1:
                ob, rew, done, info = self.env.step(self.curac, want_render=False)
            else:
                ob, rew, done, info = self.env.step(self.curac)
            totrew += rew
            if done: break
        return ob, totrew, done, info

    def seed(self, s):
        self.rng.seed(s)
        
def down_sampeling(img, N):
    img = cv2.resize(img, 
                     dsize=(int(img.shape[1]/N), int(img.shape[0]/N)), 
                     interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
            img = img[:,:, None]
    return img
    
def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def contour(state):
    edges = cv2.Canny(state, 100, 150)
    bitwise = cv2.bitwise_not(edges)
    return bitwise

class Processing_wapper(gym.Wrapper):
    def __init__(self, env, N):
        self.n = N
        super(Processing_wapper, self).__init__(env)
        old_img1 = env.observation_space
        new_shape = (1, old_img1.shape[0]//N//2, old_img1.shape[1]//N)
        # self.observation_space = spaces.Box(low=0, high=255, shape = new_shape, dtype=np.uint8)
        self.observation_space =  spaces.Dict({'main': spaces.Box(low=0, high=255, shape = new_shape, dtype=np.uint8),
                                                'speed': spaces.Box(low=0, high=800, shape = ([1]), dtype=np.float16)})
    def reset(self):
        state = self.env.reset()
        
        state1, state2 = np.vsplit(state, 2)
        state1 = down_sampeling(rgb2gray(state1), self.n)
        state1 = np.array(state1.astype(np.uint8)).transpose(2, 0, 1)
        return {'main':state1, 'speed':np.array([0])}
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        state1, state2 = np.vsplit(state, 2)
        state1 = down_sampeling(rgb2gray(state1), self.n)
        state1 = np.array(state1.astype(np.uint8)).transpose(2, 0, 1)
        
        return {'main':state1, 'speed':info['speed']}, reward, done, info
    
class SuperMarioKartDiscretizer(gym.ActionWrapper):
    def __init__(self, env):
        super(SuperMarioKartDiscretizer, self).__init__(env)
        buttons = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
        _actions = [False for i in buttons]
        actions = [['B'], ['LEFT'], ['RIGHT'], ['B', 'LEFT'], ['B', 'RIGHT'],
                   ['B', 'R'], ['B', 'LEFT', 'R'], ['B', 'RIGHT', 'R']]
    
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))
#         self.action_space = gym.spaces.MultiBinary(len(buttons))
        
    def action(self, a):
        #return a
#         print(a)
#         print(self._actions[a].copy())
        return self._actions[a].copy()
    
    
class TensorboardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_PPO_model')
        self.best_mean_reward = -np.inf
        
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            
            if len(x) > 0:
                mean_reward = np.mean(y[-5:])
                if self.verbose > 0:
                    self.logger.record('Best mean reward', self.best_mean_reward)
                    self.logger.record('Last mean reward per episode', mean_reward)
                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)
                    
        return True