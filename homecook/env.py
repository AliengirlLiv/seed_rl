from seed_rl import embodied
import numpy as np
import gym

from PIL import Image, ImageFont, ImageDraw


class HomeCook(embodied.Env):

  def __init__(
    self,
    task,
    mode="train",
    size=(64, 64),
    # env config
    max_steps=100,
    num_trashobjs=2,
    num_trashcans=2,
    p_teleport=0.1,
    p_unsafe=0.1,
    # lang wrapper config
    language_obs="embeds",
    repeat_task_every=20,
    preread_max=-1,
    p_language=0.2,
    lang_types=["task"],
  ):
    assert task in ("longcleanup")
    assert language_obs in ("strings", "embeds", "token_embeds", "sentence_embeds")
    from minigrid.homegridv2 import HomeCook
    from minigrid.homegridv2_wrappers import (
      PickupTask, LongEpCleanup, Preread, RepeatLast,
      TeleportedDesc
    )
    from minigrid.homegrid_token_wrappers import (
      MultitaskWrapper,
      LanguageWrapper
    )
    from minigrid.custom_wrappers import (
      RGBImgPartialObsWrapper,
      FilterObsWrapper,
      Gym26Wrapper
    )
    from seed_rl import from_gym

    print("HomeCook config:")
    print("  task: ", task)
    print("  types: ", lang_types)
    print("  preread_max: ", preread_max)
    print("  underlying max_steps: ", max_steps)
    print("  repeat_task_every: ", repeat_task_every)

    env = HomeCook(p_object_in_hand=0.0,
                   max_steps=max_steps,
                   num_trashobjs=num_trashobjs,
                   num_trashcans=num_trashcans,
                   p_teleport=p_teleport,
                   p_unsafe=p_unsafe)
    env = RGBImgPartialObsWrapper(env)
    env = FilterObsWrapper(env, ["image"])
    self.language_obs = language_obs
    assert task == "longcleanup" and language_obs in ("token_embeds", "sentence_embeds")
    if language_obs in ["token_embeds", "sentence_embeds"]:
      env = MultitaskWrapper(env)
      env = LanguageWrapper(
        env,
        preread_max=preread_max,
        repeat_task_every=repeat_task_every,
        p_language=p_language,
        lang_types=lang_types,
        return_sentence_embed=language_obs == "sentence_embeds",
      )
    elif language_obs == "embeds":
      env = LongEpCleanup(
        env,
        language_obs=language_obs)
    else:
      raise NotImplementedError()

    env = Gym26Wrapper(env)
    self._env = env
    self.observation_space = env.observation_space
    if self.language_obs == "sentence_embeds":
      del self.observation_space.spaces["token"]
      del self.observation_space.spaces["token_embed"]
    self.action_space = self._env.action_space
    self.wrappers = [
      from_gym.FromGym,
      lambda e: embodied.wrappers.ResizeImage(e, size),
    ]

  def reset(self):
    obs = self._env.reset()
    del obs["log_language_info"]
    obs["token"] = np.array(obs["token"], dtype=np.uint32)
    if self.language_obs == "sentence_embeds":
      obs["sentence_embed"] = obs["sentence_embed"].astype(np.float32)
      del obs["token"]
      del obs["token_embed"]
    return obs

  def step(self, action):
    obs, rew, done, info = self._env.step(action)
    del obs["log_language_info"]
    obs["token"] = np.array(obs["token"], dtype=np.uint32)
    if self.language_obs == "sentence_embeds":
      obs["sentence_embed"] = obs["sentence_embed"].astype(np.float32)
      del obs["token"]
      del obs["token_embed"]
    return obs, rew, done, info

  def render(self):
    return self._env.render(mode="rgb_array")

  def render_with_text(self, text):
    img = self._env.render(mode="rgb_array")
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, (0, 0, 0))
    draw.text((0, 45), "Action: {}".format(self._env.prev_action), (0, 0, 0))
    img = np.asarray(img)
    return img


def create_environment(*args, **kwargs):
    env = HomeCook(*args, **kwargs)
    return env
