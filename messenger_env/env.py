from seed_rl import embodied
import numpy as np
import gym
from gym import spaces
import os
import pickle
from PIL import Image, ImageDraw

class Messenger(embodied.Env):

  def __init__(
    self,
    task,
    mode,
    size=(16, 16),
    message_prob=0.2,
    language_obs="token_embeds",
    length=64,
    load_embeddings=True,
    use_overfit=False,
  ):

    print((f"Messenger config: "
           f"  {task} {mode}\n"
           f"  length: {length}\n"
           f"  language obs: {language_obs}\n"
           f"  message_p: {message_prob}\n"
           f"  use_overfit: {use_overfit}\n"))
    assert task in ("s1", "s2", "s3")
    assert mode in ("train", "eval")
    assert language_obs in ("strings", "token_embeds",
                            "token_embeds_all")
    assert language_obs == "token_embeds", "currently only token_embeds supported"
    import messenger
    
    from messenger.envs.stage_one import StageOne
    from messenger.envs.stage_two import StageTwo
    from messenger.envs.stage_three import StageThree
    from messenger.envs.wrappers import TwoEnvWrapper
    from messenger.envs.config import STATE_HEIGHT, STATE_WIDTH
    from seed_rl import from_gym

    if task == "s1":
      mmode = "train" if mode == "train" else "val"
      if mode == "train":
        self._env = TwoEnvWrapper(
          stage=1,
          split_1='train-mc',
          split_2='train-sc',
          prob_env_1=0.75,
          message_prob=message_prob,
          use_overfit=use_overfit,
        )
      else:
        self._env = StageOne(split="val", message_prob=message_prob)
    elif task == "s2":
      if mode == "train":
        # Victor's two stage config; original one is different?
        self._env = TwoEnvWrapper(
          stage=2,
          split_1='train-sc',
          split_2='train-mc'
        )
      else:
        self._env = StageTwo(split='val')
    elif task == "s3":
      if mode == "train":
        self._env = TwoEnvWrapper(
          stage=3,
          split_1='train-mc',
          split_2='train-sc',
          prob_env_1=0.75,
        )
      else:
        self._env = StageThree(split='val')

    # Wrappers
    self.wrappers = [
      from_gym.FromGym,
      lambda e: embodied.wrappers.PadImage(e, "image", size),
      lambda e: embodied.wrappers.ReadFirst(e, duration=readfirst)
    ]

    self.language_obs = language_obs
    if load_embeddings:
      with open(f"{os.path.dirname(__file__)}/data/messenger_embeds.pkl", "rb") as f:
        self.token_cache, self.embed_cache = pickle.load(f)
      self.empty_token_id = self.token_cache["<pad>"]
      self.empty_token_embed = self.embed_cache["<pad>"]
    else:
      self._init_models()

    self.manual = None
    self.tokens = []
    self.current_sentence = None
    self._step = 0
    self._init_obs = None
    self.length = length
    self.reading = False
    self.read_step = 0
    self.max_token_seqlen = 36

    self.n_entities = 17
    self.grid_size = (STATE_HEIGHT, STATE_WIDTH)

  def _init_models(self):
    self.token_cache = {}
    self.embed_cache = {}
    if self.language_obs == "token_embeds" \
        or self.language_obs == "token_embeds_all":
      from transformers import T5Tokenizer
      self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
      self.empty_token_id = self.tokenizer.pad_token_id
      from transformers import T5EncoderModel
      self.encoder = T5EncoderModel.from_pretrained("t5-small")
      self.empty_token_embed = self._embed("<pad>")[0][0]

  def _symbolic_to_multihot(self, obs):
    # (h, w, 2)
    layers = np.concatenate((obs["entities"], obs["avatar"]),
                            axis=-1).astype(int)
    new_ob = np.maximum.reduce([np.eye(self.n_entities)[layers[..., i]] for i
                                in range(layers.shape[-1])])
    new_ob[:, :, 0] = 0
    assert new_ob.shape == self.observation_space["image"].shape
    return new_ob.astype(np.float32)

  @property
  def observation_space(self):
    obs_space = {
      "image": spaces.Box(
        low=0,
        high=1,
        shape=(*self.grid_size, self.n_entities),
      ),
      "token": spaces.Box(
          0, 32100,
          shape=(1,),
          dtype=np.uint32),
      "token_embed": spaces.Box(
          -np.inf, np.inf,
          shape=(512,),
          dtype=np.float32),
      "is_read_step": spaces.Box(
        low=np.array(False),
        high=np.array(True),
        shape=(),
        dtype=bool,
      )
      }
    return obs_space

  @property
  def action_space(self):
    return self._env.action_space

  def _embed(self, string):
    assert self.language_obs == "token_embeds" or \
      self.language_obs == "token_embeds_all"
    if f"{string}_{self.max_token_seqlen}" not in self.token_cache \
        or string not in self.embed_cache:
      pad_cfg = {} if self.language_obs != "token_embeds_all" else {
        "padding": "max_length",
        "max_length": self.max_token_seqlen
      }
      print(string)
      tokens = self.tokenizer(string, return_tensors="pt",
                              add_special_tokens=True,  # add </s> separators
                              **pad_cfg)
      import torch
      with torch.no_grad():
        # (seq, dim)
        embeds = self.encoder(**tokens).last_hidden_state.squeeze(0)
      self.embed_cache[string] = embeds.cpu().numpy()
      self.token_cache[f"{string}_{max_len}"] = {
        k: v.squeeze(0).cpu().numpy() for k, v in tokens
      }
    return (
      self.embed_cache[string],
      self.token_cache[f"{string}_{self.max_token_seqlen}"]
    )

  def _tokenize_max_len(self, string, max_len):
    if f"{string}_{max_len}" in self.token_cache:
      return self.token_cache[string]
    tokens = self.tokenizer(string, add_special_tokens=False,
                            padding="max_length", max_length=max_len,
                            return_tensors="np")
    tokens = {k: v.squeeze(0) for k, v in tokens.items()}
    self.token_cache[string] = tokens
    return tokens

  def reset(self):
    self._step = 0
    self.read_step = 0
    obs, self.manual_sentences = self._env.reset()
    self.manual = "</s>" + "</s>".join([x.strip() for x in self.manual_sentences])
    if self.language_obs == "strings":
      obs["language_info"] = self.manual
    elif self.language_obs == "token_embeds":
      self.token_embeds = []
      self.tokens = [] # for logging
      for sent in self.manual_sentences:
        es, ts = self._embed(sent)
        # Remove padding
        es = es[ts["attention_mask"].astype(bool)]
        ts = ts["input_ids"][ts["attention_mask"].astype(bool)]
        self.token_embeds += [tok_e for tok_e in es]
        self.tokens += [tok for tok in ts]
      assert len(self.token_embeds) == len(self.tokens)
      obs.update({
        "token": self.tokens[self.read_step],
        "token_embed": self.token_embeds[self.read_step],
      })
      self.reading = True
    elif self.language_obs == "token_embeds_all":
      self.token_embeds_all = []
      self.tokens = []
      self.full_manual_tokens = {"input_ids": [], "attention_mask": []}
      for sent in self.manual_sentences:
        es, ts = self._embed(sent)
        self.token_embeds_all.append(es)
        self.tokens.append(ts["input_ids"])
        self.full_manual_tokens["input_ids"].extend(ts["input_ids"])
        self.full_manual_tokens["attention_mask"].extend(
          ts["attention_mask"])
      self.token_embeds_all = np.array(self.token_embeds_all)
      obs.update({
        # (num_sents=3, seq, dim)
        "token_embeds_all": self.token_embeds_all,
        # For decoder: reconstruct token ids
        "language_info_input_ids": self.full_manual_tokens["input_ids"],
        "language_info_attention_mask": self.full_manual_tokens["attention_mask"],
      })
      self.reading = True
    self.read_step += 1
    obs["image"] = self._symbolic_to_multihot(obs)
    obs["is_read_step"] = self.reading
    del obs["entities"]
    del obs["avatar"]
    self._init_obs = obs
    return obs

  def step(self, action):
    if self.reading:
      obs = self._init_obs
      obs["is_read_step"] = self.reading
      if self.language_obs == "token_embeds":
        obs["token"] = self.tokens[self.read_step]
        obs["token_embed"] = self.token_embeds[self.read_step]
      elif self.language_obs == "token_embeds_all":
        obs.update({
          "token_embeds_all": self.token_embeds_all,
          "language_info_input_ids": self.full_manual_tokens["input_ids"],
          "language_info_attention_mask": self.full_manual_tokens["attention_mask"],
        })

      else:
        raise NotImplementedError()
      self.read_step += 1
      if self.language_obs == "token_embeds_all":
        if self.read_step >= 15:
          self.reading = False
          self.read_step = 0
      elif self.read_step >= len(self.tokens):
        self.reading = False
        self.read_step = 0
      return obs, 0, False, None

    self._step += 1 # don't increment step while reading
    obs, rew, done, info = self._env.step(action)
    obs["is_read_step"] = self.reading
    info = info or {}
    if self.language_obs == "strings":
      obs["language_info"] = self.manual
    elif self.language_obs == "tokens":
      obs["token"] = self.empty_token_id
    elif self.language_obs == "token_embeds":
      obs["token"] = self.empty_token_id
      obs["token_embed"] = self.empty_token_embed
    elif self.language_obs == "token_embeds_all":
      obs.update({
        "token_embeds_all": self.token_embeds_all,
        "language_info_input_ids": self.full_manual_tokens["input_ids"],
        "language_info_attention_mask": self.full_manual_tokens["attention_mask"],
      })
    obs["image"] = self._symbolic_to_multihot(obs)
    info.update({
      "entities": obs["entities"],
      "avatar": obs["avatar"],
    })
    del obs["entities"]
    del obs["avatar"]

    if self._step >= self.length:
      done = True
      rew = -1
    return obs, rew, done, None

  def make_image(self, img, ac, rew, done):
    assert len(img.shape) == 3
    assert img.shape[2] == 17
    img = img[:10, :10]  # remove padding

    idx_to_letter = {
      2: 'A', 3: 'M', 4: 'D', 5: 'B', 6: 'F', 7: 'C', 8: 'T', 9: 'H', 10: 'B',
      11: 'R', 12: 'Q', 13: 'S', 14: 'W', 15: 'a', 16: 'm',
    }
    idx_to_entity_name = {
      2: 'airplane', 3: 'mage', 4: 'dog', 5: 'bird', 6: 'fish', 7: 'scientist', 8: 'thief', 9: 'ship', 10: 'ball', 11: 'robot', 12: 'queen', 13: 'sword', 14: 'wall', 15: 'player', 16: 'player',
    }

    role_to_colors = {
      'player_with_message': 'pink',
      'player_without_message': 'orange',
      'message': 'blue',
      'enemy': 'red',
      'goal': 'green',
      'other': 'gray',
    }
    actions = ["up", "down", "left", "right", "stay", "reset"]
    scale = 30
    new_img = Image.fromarray(np.zeros((10 * scale, 10 * scale, 3), dtype=np.uint8) + 255)
    draw = ImageDraw.Draw(new_img)
    idxs = img.argmax(-1)
    for i, row in enumerate(img):
      for j, col in enumerate(row):
        if idxs[i][j] == 0: continue
        letter = idx_to_letter[idxs[i][j]]
        # x,y canvas reversed
        draw.text((j * scale, i * scale), letter, "red")
    manual = "//".join(self.manual_sentences)
    manual = manual.encode("ascii", "ignore").decode("ascii")
    chunk_size = 40
    chunks = (len(manual) // chunk_size) + 1
    manual2 = [manual[i * chunk_size:(i+1) * chunk_size] for i in range(0, chunks)]
    manual2.append(f"a {actions[ac]} r {rew} {done}")
    draw.multiline_text((0, 0), "\n".join(manual2), (0, 0, 0))
    new_img = np.asarray(new_img)
    return new_img


def create_environment(task, mode, separate_sentences, message_prob):
    env = Messenger(task, mode, separate_sentences, message_prob)
    return env
