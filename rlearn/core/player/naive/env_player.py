from .base import BaseEnvPlayer

class EnvPlayer(BaseEnvPlayer):
    """
    A simple environment player that wraps a gymnasium environment.
    """

    def __init__(self, env):
        self.env = env

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()
        
    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space
