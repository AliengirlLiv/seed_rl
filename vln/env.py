from seed_rl import embodied
import numpy as np

import sys
sys.path = ['/home/olivia/LangWorld/docker_seed/seed_rl/VLN_CE', '/home/olivia/LangWorld/docker_seed/seed_rl', '/home/olivia/LangWorld/dynalangv3'] + sys.path

from VLN_CE.vlnce_baselines.common.env_utils import (
    construct_env,
)
from VLN_CE.vlnce_baselines.config.default import get_config
from habitat_lab.habitat_baselines.common.environments import get_env_class
import os
import random
from PIL import Image, ImageFont, ImageDraw
import pickle
from gym import spaces


class VLNEnv(embodied.Env):

  def __init__(self, task=None, mode='train', size=(64, 64), length=500, use_text=True, use_depth=False, use_stored_tokens=False, 
               load_embeddings=True, dataset='train', use_expert=0, min_use_expert=0, anneal_expert_eps=0,  success_reward=0, early_stop_penalty=0, use_descriptions=False, desc_length=50, language_obs="token_embeds_all", seed=None):
    
    assert language_obs in ("token_embeds", "token_embeds_all")
    self.language_obs = language_obs

    assert mode in dataset, "Mismatched env mode and dataset"

    self._task = 'cont'
    self._size = size
    self._length = length
    self._step = 0
    self._done = False
    self._mode = mode
    self._use_text = use_text
    self._use_depth = use_depth
    self._use_stored_tokens = use_stored_tokens
    self._load_embeddings = load_embeddings
    self._use_expert = use_expert
    self._use_descriptions = use_descriptions
    self._desc_length = desc_length
    self._min_use_expert = min_use_expert
    self._anneal_expert_eps = anneal_expert_eps
    self._success_reward = success_reward
    self._early_stop_penalty = early_stop_penalty
    self.max_token_seqlen = '' # for -1
    self.read_step = 0 # What token we are currently feeding in
    self.done_first_input = False # True if we have finished feeding in the first text input, forcing agent to listen to the whole instruction
    self.cur_text_type = 'instr' # What type of text we are currently feeding in, can be 'instr' or 'desc'
    self.cur_text = '' # What text we are currently feeding in
    self._num_eps = 0

    if seed is None:
      seed = 42
    assert self._desc_length < self._length
    
    config_opts = ['TASK_CONFIG.DATASET.SPLIT', dataset, 'TASK_CONFIG.TASK.NDTW.SPLIT', dataset, 'TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE', mode == 'train', 'TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE', mode=='train']
    if mode == 'test':
      config_opts.extend(['TASK_CONFIG.TASK.SENSORS', ['INSTRUCTION_SENSOR']])
      config_opts.extend(['TASK_CONFIG.TASK.MEASUREMENTS', ['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL', 'ORACLE_SUCCESS', 'NDTW', 'PATH_LENGTH']])
    # TODO: clean up config pathing
    self.config =  get_config(os.path.dirname(os.path.realpath(__file__)) + '/vln.yaml', opts=config_opts)
    self._env = construct_env(
            self.config,
            get_env_class(self.config.ENV_NAME)
    )

    
    if not use_stored_tokens: # Don't use stored GLoVE embeddings from VLN-CE
      if load_embeddings:
        if self.language_obs == 'token_embeds':
          with open(f"{os.path.dirname(__file__)}/data/vln_embeds_t5.pkl", "rb") as f:
           self.token_cache, self.embed_cache = pickle.load(f)
          self.empty_token_id = self.token_cache["<pad>"]
          self.empty_token_embed = self.embed_cache["<pad>"]
        elif self.language_obs == 'token_embeds_all':
          self.token_cache = {}
          with open(f"{os.path.dirname(__file__)}/data/vln_embeds_st.pkl", "rb") as f:
            self.embed_cache = pickle.load(f)
        else:
          raise NotImplementedError
      else:
        self._init_models()

    # if not use_stored_tokens:
    #   from sentence_transformers import SentenceTransformer
    #   os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #   self.lm = SentenceTransformer("all-distilroberta-v1").eval()
    #   self.lm_embed_size = 768
    #   if self._use_descriptions == 'embed_concat':
    #     self.lm_embed_size *= 2
    #   self.cache = {}

  def _init_models(self):
    self.token_cache = {}
    self.embed_cache = {}
    if self.language_obs == "token_embeds":
      from transformers import T5Tokenizer
      self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
      self.empty_token_id = self.tokenizer.pad_token_id
      from transformers import T5EncoderModel
      self.encoder = T5EncoderModel.from_pretrained("t5-small")
      self.empty_token_embed = self._embed("<pad>")[0][0]    
    elif self.language_obs == "token_embeds_all":
      from sentence_transformers import SentenceTransformer
      self.encoder = SentenceTransformer("all-distilroberta-v1").eval()
 
  @property
  def observation_space(self):
    sp = self._env.observation_space
    new_space = {}
    # resize image
    new_space['image'] = spaces.Box(dtype=sp['rgb'].dtype, shape=self._size + (3,), low=np.zeros(self._size + (3,), dtype=np.int8), high=255 * np.ones(self._size + (3,), dtype=np.int8))
    if self._use_depth:
      new_space['depth'] = new_space['image']
    if self._use_text:
      # use one field for instructions or description text
      if self._use_stored_tokens: 
        new_space['token_embeds_all'] = spaces.Box(np.int, (200,)) # TODO: fix this arbitrary max length
      else:
        if self.language_obs == "token_embeds":
          new_space.update({
            "token": spaces.Box(
          0, 32100,
          shape=(),
          dtype=np.int64),
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
          })
        elif self.language_obs == "token_embeds_all":
          new_space.update({
            "token_embeds_all": spaces.Box(
              low=-np.inf, high=np.inf,
              shape=(768,),
              dtype=np.float32),
          })
        else:
          raise NotImplementedError(self.language_obs)


    return new_space
  @property
  def action_space(self):
    self._disc_act_space = ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']
    return spaces.Discrete(len(self._disc_act_space))
  
  def _embed(self, string):
    assert self.language_obs == "token_embeds" or \
      self.language_obs == "token_embeds_all"

    string = string.strip().replace('\n', ' ').replace('\r', '')
    
    if string not in self.embed_cache:
      print('Missing from cache!! String:', string)
      if self.language_obs == 'token_embeds':
        pad_cfg = {} if self.language_obs != "token_embeds_all" else {
          "padding": "max_length",
          "max_length": self.max_token_seqlen
        }
        # print(string)
        tokens = self.tokenizer(string, return_tensors="pt",
                                add_special_tokens=True,  # add </s> separators
                                **pad_cfg)
        import torch
        with torch.no_grad():
          # (seq, dim)
          embeds = self.encoder(**tokens).last_hidden_state.squeeze(0)
        self.embed_cache[string] = embeds.cpu().numpy()
        self.token_cache[f"{string}{self.max_token_seqlen}"] = {
          k: v.squeeze(0).cpu().numpy() for k, v in tokens}
      elif self.language_obs == 'token_embeds_all':
        embeds = self.encoder.encode([string], convert_to_numpy=True)
        self.embed_cache[string] = embeds

    if self.language_obs == 'token_embeds_all':
      return self.embed_cache[string]
    else:
      return (
        self.embed_cache[string],
        self.token_cache[f"{string}{self.max_token_seqlen}"]
      )

  def convert_depth(self, depth):
    # normalize and clip depth images
    depth = (np.clip(depth, 0, 5.0) / 5.0 * 255).astype(np.uint8) # Clip to 5m, convert to uint8)
    depth = np.repeat(depth, 3, axis=-1)

    depth = Image.fromarray(depth)
    depth = depth.resize(self._size)
    depth = np.asarray(depth, dtype=np.uint8)

    return depth

  def reset(self):
    self._num_eps += 1 
    self._step = 0
    self.read_step = 0
    self.cur_text = ''
    self.cur_text_type = 'instr'
    self.tokens = [] # for logging
    self._done = False
    self.done_first_input = False
    ob = self._env.reset()
    self.prev_env_ob = ob

    if self._num_eps < self._anneal_expert_eps:
      self._expert_ep = np.random.rand() < self._use_expert - (self._use_expert - self._min_use_expert) / self._anneal_expert_eps * self._num_eps
    elif self._min_use_expert == self._use_expert: 
      self._expert_ep = np.random.rand() < self._use_expert
    else:
      self._expert_ep = np.random.rand() < self._min_use_expert

    ob = self.format_obs(ob)
    ob["is_read_step"] = not self.done_first_input
    ob[f'log_{self._mode}_success'] = 0
    ob[f'log_{self._mode}_pl_success'] = 0
    ob[f'log_{self._mode}_oracle_success'] = 0

    if self._expert_ep:
      # need to get infos to get gt_actions
      self.next_expert_ac = self.prev_env_ob['shortest_path_sensor'][0]
    
    return ob

  def step(self, action):
    
    if self.done_first_input:
      self._step += 1
      ob, rew, dones, infos = self._env.step(action)
      if action == 0:
        if infos['success']: 
          rew = self._success_reward
        else:
          rew = self._early_stop_penalty #stop action too early
      

      self.prev_env_ob = ob
    else:
      # need to listen to instruction first
      ob = self.prev_env_ob
      rew = 0
      dones = False
      infos = {'success': 0, 'spl': 0, 'oracle_success': 0}

    
    log_traj_id = ob['instruction']['trajectory_id']
    ob = self.format_obs(ob)
    if self._expert_ep:
      self.next_expert_ac = self.prev_env_ob['shortest_path_sensor'][0]
    self._done = (self._step >= self._length) or dones
    ob["is_read_step"] = not self.done_first_input
    ob[f'log_{self._mode}_success'] = infos['success']
    ob[f'log_{self._mode}_pl_success'] = infos['spl']
    ob[f'log_{self._mode}_oracle_success'] = infos['oracle_success']
    return ob, rew, self._done, None

  def embed_language(self, lang):
    if lang in self.cache:
      return self.cache[lang]
    else:
      embed = self.lm.encode([lang], convert_to_numpy=True)[0]
      self.cache[lang] = embed
      return embed
    
  def get_embed_text(self, ob): 
    if (self.language_obs == "token_embeds" and len(self.tokens) > 0 and self.read_step >= len(self.tokens)) or (self.language_obs == 'token_embeds_all'):
      self.read_step = 0
      self.done_first_input = True
      if self._use_descriptions  and len(ob['descriptions']) > 1 and self._step > 0:
        self.cur_text_type = 'instr' if self.cur_text_type == 'desc' else 'desc'
      else:
        self.cur_text_type = 'instr'

    if self.read_step == 0:
      # sample new text to feed in
      if self.cur_text_type == 'instr':
        self.cur_text = ob['instruction']['text']
      elif self.cur_text_type == 'desc':
        self.cur_text = random.choice(ob['descriptions'])
      else:
        raise NotImplementedError
      self.token_embeds = []
      self.tokens = [] # for logging

      if self.language_obs == "token_embeds":
        # Remove padding 
        es, ts = self._embed(self.cur_text) # embed sentence
        # es = es[ts["attention_mask"].astype(bool)]
        # ts = ts["input_ids"][ts["attention_mask"].astype(bool)]
        self.token_embeds = [tok_e for tok_e in es]
        self.tokens = [tok for tok in ts]
        assert len(self.token_embeds) == len(self.tokens)
      elif self.language_obs == 'token_embeds_all':
        self.token_embeds = self._embed(self.cur_text) # embed sentence

    if self.language_obs == "token_embeds":
      # print(self.cur_text, self.read_step, self.tokens[self.read_step])
      new_ob = {
          "token": self.tokens[self.read_step],
          "token_embed": self.token_embeds[self.read_step],
        }
      self.read_step += 1
    elif self.language_obs == 'token_embeds_all':
      # print(self.cur_text)
      new_ob = {
          "token_embeds_all": self.token_embeds,
        }

    return new_ob
      


    # if self._use_descriptions == 'pre' and self._step < self._desc_length:
    #   text = random.choice(ob['descriptions'])
    #   return self.embed_language(text), text
    # elif self._use_descriptions == 'concat_embed':
    #   text = random.choice(ob['descriptions']) + ' ' + ob['instruction']['text']
    #   return self.embed_language(text), text
    # elif self._use_descriptions == 'embed_concat':
    #   inst_text = ob['instruction']['text']
    #   desc_text = random.choice(ob['descriptions'])
    #   return np.concatenate([self.embed_language(inst_text), self.embed_language(desc_text)], axis=-1),  inst_text + ' ' + desc_text
    # return self.embed_language(ob['instruction']['text']), ob['instruction']['text']

  def format_obs(self, ob):
    new_ob = {}
    img = Image.fromarray(ob['rgb'])
    img = img.resize(self._size)
    new_ob['image'] =  np.asarray(img, dtype=np.uint8)
    if self._use_depth:
      new_ob['depth'] = self.convert_depth(ob['depth'])
    
    if self._use_text:
      if self._use_stored_tokens:
        self.cur_text = ob['instruction']['text']
      else:
        new_ob.update(self.get_embed_text(ob))
    
    return new_ob
  
  def render_with_text(self, ob, instr_text, traj_id, ac):
    img = self._env.render()
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    # Define the maximum width of the text
    max_width = 256

    # Calculate the height of the text
    instr_text = 'Instruction: ' + instr_text
    instr_text = instr_text.encode("ascii", "ignore")
    instr_text = instr_text.decode()
    text_width, text_height = draw.textsize(instr_text)
    max_len = int((max_width / text_width) * len(instr_text))
    wrapped_text = "\n".join([instr_text[i:i+max_len] for i in range(0, len(instr_text), max_len)])

    draw.text((0, 0), 'Trajectory ID: {}, Mode: {}'.format(traj_id, self._mode), (0, 0, 0))
    draw.text((0, 15), "Action: {}".format(ac), (0, 0, 0))
    draw.multiline_text((0, 30), wrapped_text, fill=(0, 0, 0))
    img = np.asarray(img).copy()

    # annotate videos
    if ob[f'log_{self._mode}_success']:
      img[:5, :, 1] =  255
    if ob[f'log_{self._mode}_oracle_success']:
      img[:5, :, 2] =  255
    
    img = np.clip(img, 0, 255)
    return img


def create_environment(dataset, language_obs, use_descriptions, use_depth, use_stored_tokens, size, mode, use_expert, min_use_expert, anneal_expert_eps, success_reward, early_stop_penalty):
    env = VLNEnv(dataset=dataset, language_obs=language_obs, use_descriptions=use_descriptions, use_depth=use_depth, use_stored_tokens=use_stored_tokens, size=size, mode=mode, use_expert=use_expert, min_use_expert=min_use_expert, anneal_expert_eps=anneal_expert_eps, success_reward=success_reward, early_stop_penalty=early_stop_penalty)
    return env
