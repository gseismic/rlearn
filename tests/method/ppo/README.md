
# PPO 训练总结

## 'HalfCheetah-v4' 
文件：`test_ppo_naive_continous.py`
### 总结 
- 'rpo_alpha': None, # 0.5, ## 非None(0.5) 会导致clipfracs过高 0.7+
- num_envs = 1 时导致return突然下降后回归，num_envs = 6 时，return稳定
- 多环境训练好像同等步数运行更快，reward波动更小，增长更慢(增长斜率更小)
- 环境越多，同等步数熵越大，vloss更小，训练总步数需要的更多？
- 环境越多，
- 在归一化的advantages尺度下，使用固定的绝对Clip ε就显得更加合理了 
- 似乎环境越多，clipfracs越低, 20个环境0.001, 6个0.12
