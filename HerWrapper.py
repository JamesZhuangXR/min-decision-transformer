import gym, d4rl, torch
import numpy as np

class HerWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.env.reset()
        target_dim = np.array(self._target).shape[0]
        origin_space = self.env.observation_space
        low = np.concatenate([origin_space.low, -np.inf * np.ones(target_dim)], axis=-1)
        high = np.concatenate([origin_space.high, np.inf * np.ones(target_dim)], axis=-1)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=origin_space.dtype)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        tgt = self._target
        obs = np.concatenate([obs, tgt], axis = -1)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        tgt = self._target
        obs = np.concatenate([obs, tgt], axis = -1)
        # rew = rew - 1.0
        return obs, rew, done, info

    def compute_reward(self, achieved_goal, goal, info):
        return -(np.linalg.norm(achieved_goal - goal, axis = -1) > 0.5).astype(np.float32)

    @property
    def _target(self):
        return self.env.get_target()

    @property
    def spec(self):
        return self.env.spec

if __name__ == '__main__':
    pass