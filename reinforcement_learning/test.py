import time
import os
from train import make_env
from sb3_contrib.ppo_recurrent import RecurrentPPO

def c16to10(self, num):
    return (num // 16)*10 + num % 16

def main():
    log_dir = './logs/'
    
    env = make_env(log_dir)
    
    model = RecurrentPPO.load(os.path.join(log_dir, 'best_PPO_model.zip'), env=env)
    state = env.reset()
    total_reward = 0
    
    while True:
        env.render()
        time.sleep(1/60)
        
        action, _ = model.predict(state)
        
        state, reward, done, info = env.step(action)
        total_reward += reward[0]
        if done:
            print('reward', total_reward)
            statu = env.reset()
            total_reward = 0

if __name__ == "__main__":
    main()