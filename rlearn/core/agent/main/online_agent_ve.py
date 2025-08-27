from abc import abstractmethod
import time
import uuid
import numpy as np
from pathlib import Path
from collections import deque
from .base_agent import BaseAgent
from ....utils.i18n import Translator
# from ....utils.exit_monitor.exit_monitor_ve import ExitMonitorVE
from ....utils.exit_monitor.exit_monitor import ExitMonitor

class OnlineAgentVE(BaseAgent):
    def __init__(self, env=None, config=None, logger=None, seed=None):
        super().__init__(env, config, logger, seed)
    
    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass
    
    def before_learn(self, *args, **kwargs):
        pass
    
    def before_episode(self, *args, **kwargs):
        pass
    
    def after_episode(self, epoch, episode_reward=None, *args, **kwargs):
        pass
    
    def after_learn(self, *args, **kwargs):
        pass
    
    def learn(self, 
              max_epochs,
              steps_per_epoch, 
              max_total_steps=None, 
              max_runtime=None,
              max_episodes=None,
              target_episode_reward=None,
              reward_window_size=100,
              min_reward_threshold=None,
              max_episodes_without_improvement=None,
              verbose_freq=10,
              checkpoint_freq=None,
              checkpoint_dir='checkpoints',
              final_model_name=None,
              final_model_dir='final_models'):
        """
        avg_reward := avg reward of all environments in recent reward_window_size episodes
        exit if any:
            (1) avg_reward >= reward_threshold and min_reward_threshold <= avg_reward
            (2) avg_reward >= max_reward_threshold
            (3) avg_reward >= best_avg_reward + improvement_threshold
            (4) avg_reward > best_avg_reward * (1 + improvement_ratio_threshold)
        """
        if self.env is None:
            raise ValueError("Environment not set. Please call set_env() before learning.")
        exit_monitor = ExitMonitor({
            'max_total_steps': max_total_steps,
            'max_runtime': max_runtime,
            'max_episodes': max_episodes,
            'target_episode_reward': target_episode_reward,
            'reward_window_size': reward_window_size,
            'min_reward_threshold': min_reward_threshold,
            'max_episodes_without_improvement': max_episodes_without_improvement,
        })
        self.num_envs = self.env.num_envs
        self.single_action_space = self.env.single_action_space
        self.single_observation_space = self.env.single_observation_space
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        
        tr = Translator(to_lang=self.lang)
        
        # here: the only `reset`
        states, infos = self.env.reset()
        if len(states.shape) == 1:
            states = states.reshape(-1, 1)
        # print(states)
        self.before_learn(states, infos, max_epochs=max_epochs, steps_per_epoch=steps_per_epoch)
        total_steps = 0
        start_time = time.time()
        
        cur_episode_ends = np.zeros(self.num_envs, dtype=bool) # 当前episode是否结束
        cur_episode_acc_rewards = np.zeros(self.num_envs) # 当前最新累积的episode reward
        cur_episode_acc_lengths = np.zeros(self.num_envs) # 当前最新累积的episode length

        should_exit_program = False 
        for epoch in range(max_epochs):
            if should_exit_program: 
                break 
            self.before_episode(epoch=epoch)
            # 不能在此reset，因为 steps_per_epoch不是真正的结束
            for epoch_step in range(steps_per_epoch):
                actions = self.select_action(states, epoch_step=epoch_step)
                (next_obs, rewards, terminates, truncates, infos) = self.env.step(actions)
                # TODO: terminates为 True才应看为done 
                dones = np.logical_or(terminates, truncates) 
                cur_episode_ends = np.logical_or(terminates, truncates) 
                assert cur_episode_ends.shape == (self.num_envs, )
                
                # 只有episode结束，且next_obs为None时，才用全零状态代替
                next_obs = np.array([
                    np.zeros(self.single_observation_space.shape)
                    if done and obs is None else obs 
                    for obs, done in zip(next_obs, dones)
                ])
                if len(next_obs.shape) == 1:
                    next_obs = next_obs.reshape(-1, 1)
                
                # new 
                cur_episode_acc_rewards += np.array(rewards)
                cur_episode_acc_lengths += 1 
                
                total_steps += self.num_envs # 环境步数 
                
                self.step(next_obs, rewards, terminates, truncates, infos,
                          epoch=epoch, epoch_step=epoch_step)
                
                states = next_obs
                
                # 因为一次时刻可能多个环境完成 
                if epoch_step == steps_per_epoch - 1:
                    should_exit_learning, episode_info = self.after_episode(epoch=epoch)
                else:
                    should_exit_learning = False
                    episode_info = None
                
                if np.sum(cur_episode_ends) > 0:
                    # 有新episode 
                    # 只用到了cur_episode_ends真的数据  
                    should_exit, exit_reason = exit_monitor.should_exit(
                        total_steps,
                        cur_episode_ends,
                        cur_episode_acc_rewards, 
                        cur_episode_acc_lengths
                    ) 

                    if should_exit:
                        self.logger.info(f"{tr('exit_reason')}: {tr(exit_reason)}")
                        should_exit_program = True
                        break 
                                  
                cur_episode_acc_rewards[cur_episode_ends] = 0 
                cur_episode_acc_lengths[cur_episode_ends] = 0 

                if should_exit_learning: 
                    self.logger.info(f'Early stopping: {str(episode_info)}')
                    should_exit_program = True 
                    break
            
            if checkpoint_freq and exit_monitor.episode_count % checkpoint_freq == 0:
                checkpoint_dir = checkpoint_dir or 'checkpoints'
                checkpoint_file = Path(checkpoint_dir) / f'checkpoint_episode_{exit_monitor.episode_count}.pth'
                # in case of user overridding save-method
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                self.save_checkpoint(str(checkpoint_file))
                self.logger.info(tr('checkpoint_saved') + f': {checkpoint_file}')
                
        self.after_learn()
        
        end_time = time.time()
        training_duration = end_time - start_time
        if not final_model_dir:
            final_model_dir = 'final_model'
        if not final_model_name:
            final_model_name = f'final_model_{uuid.uuid4().hex[:8]}.pth'
        
        final_model_file = Path(final_model_dir) / final_model_name
        final_model_file.parent.mkdir(parents=True, exist_ok=True)
        self.save(str(final_model_file))
        self.logger.info(tr('final_model_saved') + f': {final_model_file}')
        
        end_time = time.time()
        training_duration = end_time - start_time

        learning_info = {
            'total_episode': exit_monitor.episode_count,
            'total_steps': total_steps,
            'training_duration': training_duration,
            'exit_reason': exit_reason,
            'final_model_file': final_model_file,
            'best_avg_reward': exit_monitor.best_avg_reward,
        }

        return learning_info
    
    @abstractmethod
    def select_action(self, states, epoch_step, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def step(self, next_obs, rewards, terminates, truncates, info,
              epoch, epoch_step):
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, state, deterministic=False):
        raise NotImplementedError()
    
    @abstractmethod 
    def model_dict(self):
        raise NotImplementedError()
    
    @abstractmethod
    def load_model_dict(self, model_dict):
        raise NotImplementedError()  
