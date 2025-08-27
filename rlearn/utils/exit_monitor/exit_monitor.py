import time
from collections import deque
import numpy as np

class ExitMonitor:
    def __init__(self, config):
        # 基本退出条件
        self.max_episodes = config.get('max_episodes', float('inf'))
        self.max_total_steps = config.get('max_total_steps', float('inf'))
        self.max_runtime = config.get('max_runtime', float('inf'))
        
        # 奖励相关条件
        self.reward_threshold = config.get('reward_threshold')
        self.reward_window_size = config.get('reward_window_size', 100)
        self.min_reward_threshold = config.get('min_reward_threshold')
        self.max_reward_threshold = config.get('max_reward_threshold')
        self.reward_check_freq = config.get('reward_check_freq', 10)
        
        # 性能改进检测
        self.no_improvement_threshold = config.get('no_improvement_threshold', 50)
        self.improvement_threshold = config.get('improvement_threshold')
        self.improvement_ratio_threshold = config.get('improvement_ratio_threshold')
        
        # 平滑计算选项
        self.smoothing_method = config.get('smoothing_method', 'sma')  # 'sma' 或 'ema'
        self.ema_alpha = config.get('ema_alpha', 0.2)  # EMA平滑系数
        
        # 状态跟踪
        self.start_time = time.time()
        self.recent_rewards = deque(maxlen=self.reward_window_size)
        self.total_steps = 0
        self.episode_count = 0
        
        # 性能改进跟踪
        self.best_avg_reward = float('-inf')
        self.episodes_without_improvement_abs = 0
        self.episodes_without_improvement_ratio = 0
        
        # EMA特定变量
        self.ema_value = None

    def _calculate_smoothed_reward(self):
        """计算平滑后的奖励值"""
        if not self.recent_rewards:
            return 0
            
        if self.smoothing_method == 'sma':
            # 简单移动平均
            return np.mean(self.recent_rewards)
        elif self.smoothing_method == 'ema':
            # 指数移动平均
            if self.ema_value is None:
                self.ema_value = np.mean(self.recent_rewards)
            else:
                # 更新EMA值
                self.ema_value = self.ema_alpha * self.recent_rewards[-1] + (1 - self.ema_alpha) * self.ema_value
            return self.ema_value
        else:
            raise ValueError(f"未知的平滑方法: {self.smoothing_method}")

    def should_exit(self, episode_reward, episode_length=None):
        self.episode_count += 1
        self.recent_rewards.append(episode_reward)
        
        # 正确累加步数
        if episode_length is not None:
            self.total_steps += episode_length
        else:
            self.total_steps += 1

        # 检查基本退出条件
        if self.max_episodes is not None and self.episode_count >= self.max_episodes:
            return True, "maximum_episodes_reached"

        if self.max_total_steps is not None and self.total_steps >= self.max_total_steps:
            return True, "maximum_total_steps_reached"

        current_runtime = time.time() - self.start_time
        if self.max_runtime is not None and current_runtime >= self.max_runtime:
            return True, "maximum_runtime_reached"

        # 只在有足够数据时检查奖励相关条件
        if len(self.recent_rewards) >= self.reward_window_size and \
           self.episode_count % self.reward_check_freq == 0:
            
            avg_reward = self._calculate_smoothed_reward()
            min_reward = np.min(self.recent_rewards)
            
            # 检查奖励阈值条件
            if self.reward_threshold is not None and avg_reward >= self.reward_threshold:
                if self.min_reward_threshold is None or min_reward >= self.min_reward_threshold:
                    return True, "reward_threshold_reached"
            
            if self.max_reward_threshold is not None and avg_reward > self.max_reward_threshold:
                return True, "exceeded_maximum_reward_threshold"
            
            # 检查绝对改进
            if self.improvement_threshold is not None:
                if avg_reward > self.best_avg_reward + self.improvement_threshold:
                    self.best_avg_reward = avg_reward
                    self.episodes_without_improvement_abs = 0
                else:
                    self.episodes_without_improvement_abs += self.reward_check_freq
                    if self.episodes_without_improvement_abs >= self.no_improvement_threshold:
                        return True, "no_performance_improvement_absolute"
            
            # 检查相对改进（比例）
            if self.improvement_ratio_threshold is not None and self.best_avg_reward != 0:
                improvement_ratio = abs(avg_reward - self.best_avg_reward) / abs(self.best_avg_reward)
                
                if improvement_ratio > self.improvement_ratio_threshold:
                    self.best_avg_reward = avg_reward
                    self.episodes_without_improvement_ratio = 0
                else:
                    self.episodes_without_improvement_ratio += self.reward_check_freq
                    if self.episodes_without_improvement_ratio >= self.no_improvement_threshold:
                        return True, "no_performance_improvement_ratio"

        return False, ""

    def reset(self):
        """重置监控器状态"""
        self.start_time = time.time()
        self.recent_rewards.clear()
        self.total_steps = 0
        self.episode_count = 0
        self.best_avg_reward = float('-inf')
        self.episodes_without_improvement_abs = 0
        self.episodes_without_improvement_ratio = 0
        self.ema_value = None

    def get_status(self):
        """获取当前监控器状态"""
        current_avg = self._calculate_smoothed_reward() if self.recent_rewards else 0
        return {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'runtime': time.time() - self.start_time,
            'current_avg_reward': current_avg,
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