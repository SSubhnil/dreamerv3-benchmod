import functools
import os

import mujoco

from dreamerv3 import embodied
import numpy as np


class FromDM(embodied.Env):

    def __init__(self, env, obs_key='observation', act_key='action', confounder_active=True, confounder_params=None,
                 force_mag=70, interval=100):
        self._env = env
        self.confounder_active = confounder_active
        obs_spec = self._env.observation_spec()
        act_spec = self._env.action_spec()
        self._obs_dict = isinstance(obs_spec, dict)
        self._act_dict = isinstance(act_spec, dict)
        self._obs_key = not self._obs_dict and obs_key
        self._act_key = not self._act_dict and act_key
        self._obs_empty = []
        self._done = True
        default_params = {
            'force_type': 'swelling',
            'timing': 'random',
            'force_magnitude': force_mag,
            'interval': interval}
        self.confounder_params = confounder_params or default_params

        # Initialize attributes based on confounder_params
        self.force_type = self.confounder_params['force_type']
        self.timing = self.confounder_params['timing']
        self.force_magnitude = self.confounder_params['force_magnitude']
        self.interval = self.confounder_params['interval']
        self.time_since_last_force = 0

    @functools.cached_property
    def obs_space(self):
        spec = self._env.observation_spec()
        spec = spec if self._obs_dict else {self._obs_key: spec}
        if 'reward' in spec:
            spec['obs_reward'] = spec.pop('reward')
        for key, value in spec.copy().items():
            if int(np.prod(value.shape)) == 0:
                self._obs_empty.append(key)
                del spec[key]
        return {
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
            **{k or self._obs_key: self._convert(v) for k, v in spec.items()},
        }

    @functools.cached_property
    def act_space(self):
        spec = self._env.action_spec()
        spec = spec if self._act_dict else {self._act_key: spec}
        return {
            'reset': embodied.Space(bool),
            **{k or self._act_key: self._convert(v) for k, v in spec.items()},
        }

    def step(self, action):
        action = action.copy()
        reset = action.pop('reset')
        if reset or self._done:
            time_step = self._env.reset()
        else:
            action = action if self._act_dict else action[self._act_key]

            self.apply_force()
            time_step = self._env.step(action)
        self._done = time_step.last()
        return self._obs(time_step)

    def _obs(self, time_step):
        if not time_step.first():
            assert time_step.discount in (0, 1), time_step.discount
        obs = time_step.observation
        obs = dict(obs) if self._obs_dict else {self._obs_key: obs}
        if 'reward' in obs:
            obs['obs_reward'] = obs.pop('reward')
        for key in self._obs_empty:
            del obs[key]
        return dict(
            reward=np.float32(0.0 if time_step.first() else time_step.reward),
            is_first=time_step.first(),
            is_last=time_step.last(),
            is_terminal=False if time_step.first() else time_step.discount == 0,
            **obs,
        )

    def _convert(self, space):
        if hasattr(space, 'num_values'):
            return embodied.Space(space.dtype, (), 0, space.num_values)
        elif hasattr(space, 'minimum'):
            assert np.isfinite(space.minimum).all(), space.minimum
            assert np.isfinite(space.maximum).all(), space.maximum
            return embodied.Space(
                space.dtype, space.shape, space.minimum, space.maximum)
        else:
            return embodied.Space(space.dtype, space.shape, None, None)

    """Custom function for applying unbalancing force"""

    def apply_force(self):
        if not self.confounder_active:
            return

        if self.timing == 'uniform' and self.time_since_last_force % self.interval != 0:
            return
        if self.timing == 'random' and np.random.uniform() > 0.1:  # 10% chance to apply force
            return

        # Construct the force vector
        if self.force_type == 'step':
            force = np.array([self.force_magnitude, 0, 0, 0, 0, 0])
        elif self.force_type == 'swelling':
            # Calculate the time step where the force magnitude is at its peak
            peak_time = self.interval / 2
            # Calculate the standard deviation to control thh width of the bell curve
            sigma = self.interval / 6  # Adjust as needed for the desired width

            # Calculate the force magnitude at the current time step using a Gaussian function
            time_step_normalized = (self.time_since_last_force - peak_time) / sigma
            magnitude = self.force_magnitude * np.exp(-0.5 * (time_step_normalized ** 2))
            force = np.array([magnitude, 0, 0, 0, 0, 0])

        # Apply the force
        self._env.physics.named.data.xfrc_applied['torso'][:] = force

        # Updates timing
        self.time_since_last_force += 1
        if self.timing == 'uniform' and self.time_since_last_force >= self.interval:
            self.time_since_last_force = 0  # resets timer for next cycle
