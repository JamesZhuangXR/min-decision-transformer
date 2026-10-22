import gym, d4rl, torch
import numpy as np

class RanTgtWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        idx = self.np_random.choice(len(self.env.empty_and_goal_locations))
        reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
        target_location = reset_location + self.np_random.uniform(low=-.5, high=.5, size=self.model.nq)
        self.env.set_target(target_location)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        rew =  -(np.linalg.norm(obs[:2] - self.env._target, axis=-1) > 0.5).astype(np.float32)
        info = {'is_success': rew + 1}
        # rew = rew - 1.0
        return obs, rew, done, info

    def compute_reward(self, achieved_goal, goal, info = None):
        return -(np.linalg.norm(achieved_goal - goal, axis = -1) > 0.5).astype(np.float32)

    @property
    def _target(self):
        return self.env.get_target()

if __name__ == '__main__':
    pass