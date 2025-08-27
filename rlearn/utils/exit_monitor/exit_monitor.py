import time
from collections import deque
import numpy as np
from rlearn.logger import user_logger

# "要求quantile的0.1的数值也能设..."点击查看元宝的回答
# https://yb.tencent.com/s/OoKdpaKHHaVB
class ExitMonitor: 
    def __init__(self, config):
        # 基本退出条件
        self.logger = user_logger
        self.max_episodes = config.get('max_episodes', float('inf'))
        self.max_total_steps = config.get('max_total_steps', float('inf'))
        self.max_runtime = config.get('max_runtime', float('inf'))
        
        # 奖励相关条件
        self.target_episode_reward = config.get('target_episode_reward') 
        self.reward_window_size = config.get('reward_window_size', 100) 
        self.min_reward_threshold = config.get('min_reward_threshold') 
        
        # 性能改进检测
        self.max_episodes_without_improvement = config.get('max_episodes_without_improvement', 50)
        
        # 平滑计算选项
        self.smoothing_method = config.get('smoothing_method', 'sma')  # 'sma' 或 'ema'
        self.ema_alpha = config.get('ema_alpha', 0.2)  # EMA平滑系数
        
        # 状态跟踪 
        self.start_time = time.time()
        self.recent_rewards = deque(maxlen=self.reward_window_size)
        self.recent_lengths = deque(maxlen=self.reward_window_size)
        self.total_steps = 0 
        self.episode_count = 0
        
        # 性能改进跟踪
        self.best_avg_reward = float('-inf')
        self.episodes_without_improvement = 0

        # 历史信息
        self.history_rewards = [] 
        self.history_lengths = [] 
        
        # EMA特定变量
        self.ema_value = None
        self.ema_lengths = None

    def _calculate_smoothed_reward_and_length(self):
        """计算平滑后的奖励值"""
        assert len(self.recent_rewards) == len(self.recent_lengths)
        
        if not self.recent_rewards:
            return 0, 0
            
        if self.smoothing_method == 'sma':
            # 简单移动平均
            return np.mean(self.recent_rewards), np.mean(self.recent_lengths)
        elif self.smoothing_method == 'ema':
            # 指数移动平均
            if self.ema_value is None:
                self.ema_value = np.mean(self.recent_rewards)
                self.ema_lengths = np.mean(self.recent_lengths)
            else:
                # 更新EMA值
                self.ema_value = self.ema_alpha * self.recent_rewards[-1] + (1 - self.ema_alpha) * self.ema_value
                self.ema_lengths = self.ema_alpha * self.recent_lengths[-1] + (1 - self.ema_alpha) * self.ema_lengths
            return self.ema_value, self.ema_lengths
        else:
            raise ValueError(f"未知的平滑方法: {self.smoothing_method}")

    def should_exit(self, 
                    total_steps,
                    cur_episode_ends, 
                    cur_episode_acc_rewards, 
                    cur_episode_acc_lengths):
        """
        当episode结束时，将当前episode的累积奖励和长度添加到recent_rewards和recent_lengths中
        
        cur_episode_ends: (num_envs, )
        cur_episode_acc_rewards: (num_envs, )
        cur_episode_acc_lengths: (num_envs, )
        """
        assert np.sum(cur_episode_ends) > 0
        assert cur_episode_acc_rewards.shape == cur_episode_acc_lengths.shape
        assert cur_episode_acc_rewards.shape == cur_episode_ends.shape
        
        self.total_steps = total_steps
     
        self.recent_rewards.extend(cur_episode_acc_rewards[cur_episode_ends])
        self.recent_lengths.extend(cur_episode_acc_lengths[cur_episode_ends])
        self.episode_count += np.sum(cur_episode_ends)
        self.history_rewards.extend(cur_episode_acc_rewards[cur_episode_ends])
        self.history_lengths.extend(cur_episode_acc_lengths[cur_episode_ends])
        
        # episode完成后才加入到总的步数中 
        # 检查基本退出条件
        if self.max_episodes is not None and self.episode_count >= self.max_episodes:
            return True, "maximum_episodes_reached"

        if self.max_total_steps is not None and self.total_steps >= self.max_total_steps:
            return True, "maximum_total_steps_reached"

        current_runtime = time.time() - self.start_time
        if self.max_runtime is not None and current_runtime >= self.max_runtime:
            return True, "maximum_runtime_reached"

        # 只在有足够数据时检查奖励相关条件
        if len(self.recent_rewards) >= self.reward_window_size:
            avg_reward, avg_length = self._calculate_smoothed_reward_and_length()
            min_reward = np.min(self.recent_rewards)
            
            # 记录日志
            self.logger.info(f"total_steps: {self.total_steps}, avg_reward: {avg_reward}, avg_length: {avg_length}")
            
            # 检查奖励阈值条件
            if self.target_episode_reward is not None and avg_reward >= self.target_episode_reward:
                if self.min_reward_threshold is None or min_reward >= self.min_reward_threshold:
                    return True, "reward_threshold_reached"
            
            # 更新无改进计数器
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.episodes_without_improvement = 0
            else:
                self.episodes_without_improvement += 1
                
            # 检查连续无改进条件
            if self.max_episodes_without_improvement is not None and self.episodes_without_improvement >= self.max_episodes_without_improvement:
                return True, "no_improvement_for_too_long"
            
        # 检查性能改进
        return False, "should_continue"

    def reset(self):
        """重置监控器状态""" 
        self.start_time = time.time() 
        self.recent_rewards.clear() 
        self.recent_lengths.clear() 
        self.total_steps = 0 
        self.episode_count = 0 
        self.best_avg_reward = float('-inf') 
        self.episodes_without_improvement = 0 
        self.ema_value = None
        self.ema_lengths = None
        
    def get_status(self):
        """获取当前监控器状态"""
        current_avg_reward, current_avg_length = self._calculate_smoothed_reward_and_length()
        return {
            'history_rewards': self.history_rewards, # * 
            'history_lengths': self.history_lengths, # *
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'runtime': time.time() - self.start_time,
            'current_avg_reward': current_avg_reward,
            'current_avg_length': current_avg_length,
            'window_size': self.reward_window_size,
            'best_avg_reward': self.best_avg_reward,
            'recent_rewards_size': len(self.recent_rewards),
            'smoothing_method': self.smoothing_method
        }

    def set_smoothing_method(self, method, alpha=None):
        """设置平滑计算方法"""
        if method not in ['sma', 'ema']:
            raise ValueError("平滑方法必须是 'sma' 或 'ema'")
        
        self.smoothing_method = method
        if alpha is not None and method == 'ema':
            self.ema_alpha = alpha
        # 重置EMA值
        self.ema_value = None
        self.ema_lengths = None