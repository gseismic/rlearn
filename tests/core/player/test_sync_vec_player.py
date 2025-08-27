import numpy as np
import pytest
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from rlearn.core.player.naive.sync_vec_env import SyncVecEnvPlayer, make_vec_env_player


def make_cartpole_env():
    """创建CartPole环境的工厂函数"""
    return gym.make('CartPole-v1')


class TestSyncVecEnvPlayer:
    """测试SyncVecEnvPlayer与Gymnasium SyncVectorEnv的一致性"""
    
    def test_initialization(self):
        """测试初始化"""
        num_envs = 4
        env_fns = [make_cartpole_env for _ in range(num_envs)]
        
        # 测试我们的实现
        our_vec_env = SyncVecEnvPlayer(env_fns)
        assert our_vec_env.num_envs == num_envs
        assert not our_vec_env.is_closed
        
        # 测试Gymnasium的实现
        gym_vec_env = SyncVectorEnv(env_fns)
        assert gym_vec_env.num_envs == num_envs
        
        # 比较空间
        assert our_vec_env.single_action_space == gym_vec_env.single_action_space
        assert our_vec_env.single_observation_space == gym_vec_env.single_observation_space
        
        our_vec_env.close()
        gym_vec_env.close()
    
    def test_reset_with_no_seed(self):
        """测试不带seed的reset"""
        num_envs = 3
        env_fns = [make_cartpole_env for _ in range(num_envs)]
        
        our_vec_env = SyncVecEnvPlayer(env_fns)
        gym_vec_env = SyncVectorEnv(env_fns)
        
        # 测试我们的实现
        our_obs, our_info = our_vec_env.reset()
        assert our_obs.shape == (num_envs, 4)  # CartPole有4个观测维度
        assert len(our_info['infos']) == num_envs
        
        # 测试Gymnasium的实现
        gym_obs, gym_info = gym_vec_env.reset()
        assert gym_obs.shape == (num_envs, 4)
        assert len(gym_info) == num_envs
        
        # 比较输出
        assert our_obs.shape == gym_obs.shape
        assert our_obs.dtype == gym_obs.dtype
        
        our_vec_env.close()
        gym_vec_env.close()
    
    def test_reset_with_single_seed(self):
        """测试带单个seed的reset"""
        num_envs = 3
        env_fns = [make_cartpole_env for _ in range(num_envs)]
        
        our_vec_env = SyncVecEnvPlayer(env_fns)
        gym_vec_env = SyncVectorEnv(env_fns)
        
        base_seed = 42
        
        # 测试我们的实现
        our_obs, our_info = our_vec_env.reset(seed=base_seed)
        
        # 测试Gymnasium的实现
        gym_obs, gym_info = gym_vec_env.reset(seed=base_seed)
        
        # 比较输出
        assert our_obs.shape == gym_obs.shape
        assert our_obs.dtype == gym_obs.dtype
        
        our_vec_env.close()
        gym_vec_env.close()
    
    def test_reset_with_list_seed(self):
        """测试带seed列表的reset"""
        num_envs = 3
        env_fns = [make_cartpole_env for _ in range(num_envs)]
        
        our_vec_env = SyncVecEnvPlayer(env_fns)
        gym_vec_env = SyncVectorEnv(env_fns)
        
        seeds = [42, 43, 44]
        
        # 测试我们的实现
        our_obs, our_info = our_vec_env.reset(seed=seeds)
        
        # 测试Gymnasium的实现
        gym_obs, gym_info = gym_vec_env.reset(seed=seeds)
        
        # 比较输出
        assert our_obs.shape == gym_obs.shape
        assert our_obs.dtype == gym_obs.dtype
        
        our_vec_env.close()
        gym_vec_env.close()
    
    def test_step(self):
        """测试step方法"""
        num_envs = 3
        env_fns = [make_cartpole_env for _ in range(num_envs)]
        
        our_vec_env = SyncVecEnvPlayer(env_fns)
        gym_vec_env = SyncVectorEnv(env_fns)
        
        # 重置环境
        our_vec_env.reset()
        gym_vec_env.reset()
        
        # 随机动作
        actions = np.array([0, 1, 0])  # 左、右、左
        
        # 测试我们的实现
        our_obs, our_rewards, our_terminateds, our_truncateds, our_info = our_vec_env.step(actions)
        
        # 测试Gymnasium的实现
        gym_obs, gym_rewards, gym_terminateds, gym_truncateds, gym_info = gym_vec_env.step(actions)
        
        # 比较输出
        assert our_obs.shape == gym_obs.shape
        assert our_rewards.shape == gym_rewards.shape
        assert our_terminateds.shape == gym_terminateds.shape
        assert our_truncateds.shape == gym_truncateds.shape
        assert our_obs.dtype == gym_obs.dtype
        assert our_rewards.dtype == gym_rewards.dtype
        assert our_terminateds.dtype == gym_terminateds.dtype
        assert our_truncateds.dtype == gym_truncateds.dtype
        
        our_vec_env.close()
        gym_vec_env.close()
    
    def test_auto_reset_on_termination(self):
        """测试环境终止时的自动重置"""
        num_envs = 2
        env_fns = [make_cartpole_env for _ in range(num_envs)]
        
        our_vec_env = SyncVecEnvPlayer(env_fns)
        gym_vec_env = SyncVectorEnv(env_fns)
        
        # 重置环境
        our_vec_env.reset()
        gym_vec_env.reset()
        
        # 运行直到某个环境终止
        max_steps = 1000
        for step in range(max_steps):
            actions = np.array([0, 0])  # 持续向左
            
            our_obs, our_rewards, our_terminateds, our_truncateds, our_info = our_vec_env.step(actions)
            gym_obs, gym_rewards, gym_terminateds, gym_truncateds, gym_info = gym_vec_env.step(actions)
            
            # 检查是否有环境终止
            if np.any(our_terminateds) or np.any(our_truncateds):
                print(f"Step {step}: 环境终止")
                print(f"我们的终止状态: {our_terminateds}, 截断状态: {our_truncateds}")
                print(f"Gymnasium终止状态: {gym_terminateds}, 截断状态: {gym_truncateds}")
                break
        
        # 继续运行几步，检查自动重置
        for step in range(5):
            actions = np.array([0, 0])
            our_obs, our_rewards, our_terminateds, our_truncateds, our_info = our_vec_env.step(actions)
            gym_obs, gym_rewards, gym_terminateds, gym_truncateds, gym_info = gym_vec_env.step(actions)
            
            print(f"后续步骤 {step}:")
            print(f"我们的终止状态: {our_terminateds}, 截断状态: {our_truncateds}")
            print(f"Gymnasium终止状态: {gym_terminateds}, 截断状态: {gym_truncateds}")
        
        our_vec_env.close()
        gym_vec_env.close()
    
    def test_make_vec_env_player(self):
        """测试make_vec_env_player函数"""
        num_envs = 4
        our_vec_env = make_vec_env_player(make_cartpole_env, num_envs)
        
        assert our_vec_env.num_envs == num_envs
        assert not our_vec_env.is_closed
        
        # 测试基本功能
        obs, info = our_vec_env.reset()
        assert obs.shape == (num_envs, 4)
        
        actions = np.array([0, 1, 0, 1])
        obs, rewards, terminateds, truncateds, info = our_vec_env.step(actions)
        assert obs.shape == (num_envs, 4)
        assert rewards.shape == (num_envs,)
        assert terminateds.shape == (num_envs,)
        assert truncateds.shape == (num_envs,)
        
        our_vec_env.close()
    
    def test_close(self):
        """测试close方法"""
        num_envs = 2
        env_fns = [make_cartpole_env for _ in range(num_envs)]
        
        our_vec_env = SyncVecEnvPlayer(env_fns)
        assert not our_vec_env.is_closed
        
        our_vec_env.close()
        assert our_vec_env.is_closed
        
        # 重复关闭应该不会出错
        our_vec_env.close()


if __name__ == "__main__":
    # 运行测试
    test_instance = TestSyncVecEnvPlayer()
    
    print("=== 测试初始化 ===")
    test_instance.test_initialization()
    
    print("\n=== 测试不带seed的reset ===")
    test_instance.test_reset_with_no_seed()
    
    print("\n=== 测试带单个seed的reset ===")
    test_instance.test_reset_with_single_seed()
    
    print("\n=== 测试带seed列表的reset ===")
    test_instance.test_reset_with_list_seed()
    
    print("\n=== 测试step方法 ===")
    test_instance.test_step()
    
    print("\n=== 测试make_vec_env_player函数 ===")
    test_instance.test_make_vec_env_player()
    
    print("\n=== 测试close方法 ===")
    test_instance.test_close()
    
    print("\n=== 测试自动重置 ===")
    test_instance.test_auto_reset_on_termination()
    
    print("\n所有测试完成！")
