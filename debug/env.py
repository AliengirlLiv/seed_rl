from seed_rl import embodied
import numpy as np
from gym import spaces
import random

class DebugEnv(embodied.Env):

  def __init__(
    self,
    obs_mode='random',
    done_mode='random',
    reward_mode='random',
  ):    
    self._obs_mode = obs_mode
    self._done_mode = done_mode
    self._reward_mode = reward_mode

  @property
  def observation_space(self):
    obs_space = {
      "image": spaces.Box(
        low=0,
        high=1,
        shape=(10, 10, 13),
        dtype=np.float32,
      ),
      "is_read_step": spaces.Box(
        low=np.array(False),
        high=np.array(True),
        shape=(),
        dtype=bool)
    }
    obs_space.update({
    "token": spaces.Box(
        0, 32100,
        shape=(),
        dtype=np.int64),
    "token_embed": spaces.Box(
        -np.inf, np.inf,
        shape=(512,),
        dtype=np.float32)
    })
    return obs_space

  @property
  def action_space(self):
    return spaces.Discrete(2)

  def reset(self):
    self._step = 0
    image = np.zeros((10, 10, 13), dtype=np.float32)
    token = np.array(0, dtype=np.int64)
    token_embed = np.zeros((512,), dtype=np.float32)
    if self._obs_mode == 'random':
        import random
        image = image + random.choice([0, 1])
        token = token + random.choice([0, 1])
        token_embed = token_embed + random.choice([0, 1])
    elif self._obs_mode == 'deterministic':
        image = image + [0, 1].index(self._step % 2)
        token = token + [0, 1].index(self._step % 2)
        token_embed = token_embed + [0, 1].index(self._step % 2)
    elif self._obs_mode == 'action':
        image = image
        token = token
        token_embed = token_embed
    obs = {}
    obs['image'] = image
    obs['token'] = token
    obs['token_embed'] = token_embed
    obs['is_read_step'] = False
    return obs

  def step(self, action):
    self._step += 1
    image = np.zeros((10, 10, 13), dtype=np.float32)
    token = np.array(0, dtype=np.int64)
    token_embed = np.zeros((512,), dtype=np.float32)
    if self._obs_mode == 'random':
        image = image + random.choice([0, 1])
        token = token + random.choice([0, 1])
        token_embed = token_embed + random.choice([0, 1])
    elif self._obs_mode == 'deterministic':
        image = image + [0, 1].index(self._step % 2)
        token = token + [0, 1].index(self._step % 2)
        token_embed = token_embed + [0, 1].index(self._step % 2)
    elif self._obs_mode == 'action':
        image = image
        token = token
        token_embed = token_embed
    obs = {}
    obs['image'] = image
    obs['token'] = token
    obs['token_embed'] = token_embed
    obs['is_read_step'] = False
    if self._reward_mode == 'random':
        rew = random.choice([0, 1])
    elif self._reward_mode == 'deterministic':
        rew = [0, 1].index(self._step % 2)
    elif self._reward_mode == 'action':
        rew = action
    elif self._reward_mode == 'obs':
        min_obs = min(obs['image'].flatten())
        rew = min_obs
    if self._done_mode == 'random':
        done = random.choice([True, False])
    elif self._done_mode == 'deterministic':
        done = self._step == 20
    elif self._done_mode == 'action':
        done = action == 1
    
    return obs, rew, done, None

 
def create_environment(obs_mode, done_mode, reward_mode):
    env = DebugEnv(obs_mode, done_mode, reward_mode)
    return env