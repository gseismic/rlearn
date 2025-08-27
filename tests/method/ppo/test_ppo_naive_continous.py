import gymnasium as gym
import time
import numpy as np
from rlearn.method.ppo.naive import PPOAgent
from rlearn.utils.eval_agent import eval_agent_performance
from rlearn.utils.seed import seed_all


# make reproducible
g_seed = None
seed_all(g_seed) # do NOT forget PPOAgent(.., seed=g_seed)

def make_env_basic(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

# same with cleanrl
def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env) # np.clip(action, self.action_space.low, self.action_space.high)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

def test_ppo_draft_continous():
    # env setup
    num_envs = 20 
    # env_id = 'CliffWalking-v0'
    # env_id = 'Hopper-v4' # 'Pendulum-v1'
    env_id = 'HalfCheetah-v4'
    capture_video = False
    run_name = "main"
    gamma = 0.99

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, i, capture_video, run_name, gamma) for i in range(num_envs)],
    )
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # loss: mainly vloss
    # 不能区分gamma和gae_lambda
    # tips: increate update_epochs
    # 当update_epochs很大的时候，clip_vloss可能必要
    # 如果gae_lambda*gamma很小，长步长在将降低到计算机精度以下
    gae_lambda = 0.95 
    config = {
        'learning_rate': 3e-4,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'rpo_alpha': None, # 0.5, ## 非None(0.5) 会导致clipfracs过高 0.7+
        'ent_coef': 0.001, # XXX zero
        'clip_coef': 0.2,
        'vf_coef': 0.5,
        'clip_vloss': True, # False, # # 在归一化的advantages尺度下，使用固定的绝对Clip ε就显得更加合理了
        'clip_coef_v': 0.2, # clip
        'update_epochs': 10, # 200
        'num_minibatches': 32, # minibatch_size: batch_size/num_minibatches
    }
    max_epochs = 500
    # 小步迭代: num_envs * steps_per_epoch
    # too small steps_per_epoch will make value not stable
    steps_per_epoch = 2048 # 200 # 2048

    agent = PPOAgent(envs, config=config, seed=g_seed)
    info = agent.learn(max_epochs, 
                       steps_per_epoch=steps_per_epoch,
                       reward_window_size=5,
                       verbose_freq=1)
    print(info)
    
    max_steps = steps_per_epoch
    single_env = gym.make(env_id)
    performance_stats = eval_agent_performance(agent, single_env, 
                                               num_episodes=10, 
                                               max_steps=max_steps,
                                               deterministic=True)
    for key, value in performance_stats.items():
        print(f"{key}: {value}")
        
    single_env.close()
    envs.close()
    
if __name__ == '__main__':
    test_ppo_draft_continous()
    