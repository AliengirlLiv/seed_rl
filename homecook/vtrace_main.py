# coding=utf-8
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""VTrace (IMPALA) example for Mujoco.

Warning!!! This code uses DeepMind wrappers which differ from OpenAI gym
wrappers and the results may not be comparable.
"""

# import sys
# sys.path = ['/home/jessy/olivia/docker_seed', '/seed_rl'] + sys.path


from absl import app
from absl import flags


import os
from seed_rl.agents.vtrace import learner
from seed_rl.agents.vtrace import networks
from seed_rl.common import actor
from seed_rl.common import common_flags  
from seed_rl.common import normalizer
from seed_rl.homecook import env
import tensorflow as tf


# Optimizer settings.
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate.')
# Network settings.
flags.DEFINE_integer('n_mlp_layers', 2, 'Number of MLP hidden layers.')
flags.DEFINE_integer('mlp_size', 64, 'Sizes of each of MLP hidden layer.')
flags.DEFINE_integer(
    'n_lstm_layers', 0,
    'Number of LSTM layers. LSTM layers are applied after MLP layers.')
flags.DEFINE_integer('lstm_size', 64, 'Sizes of each LSTM layer.')
flags.DEFINE_bool('normalize_observations', False, 'Whether to normalize'
                  'observations by subtracting mean and dividing by stddev.')
# Environment settings.
flags.DEFINE_integer('max_steps', 100, 'Number of steps per episode.')
flags.DEFINE_integer('num_trashobjs', 2, 'Number of trash objects.')
flags.DEFINE_integer('num_trashcans', 2, 'Number of trash cans.')
flags.DEFINE_float('p_teleport', 0.1, 'Probability of teleportation.')
flags.DEFINE_float('p_unsafe', 0.1, 'Probability of unsafe.')
flags.DEFINE_integer('repeat_task_every', 20, 'Repeat task every')
flags.DEFINE_integer('preread_max', -1, 'Preread max.')
flags.DEFINE_float('p_language', 0.2, 'p_language')
flags.DEFINE_list('lang_types', ['task'], 'Language types.')
flags.DEFINE_enum('lang_key', 'token', ['token', 'token_embed'], 'Language key.')


FLAGS = flags.FLAGS


def create_agent(action_space, env_observation_space,
                 unused_parametric_action_distribution):
  return networks.ImpalaDeep(action_space.n, vocab_size=env_observation_space['token'].high + 1, lang_key=FLAGS.lang_key)


def create_optimizer(unused_final_iteration):
  learning_rate_fn = lambda iteration: FLAGS.learning_rate
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  return optimizer, learning_rate_fn


def main(argv):
  create_environment = lambda task, config: env.create_environment(
    task='longcleanup',
    mode='train',
    size=(64, 64),
    # env config
    max_steps=FLAGS.max_steps,
    num_trashobjs=FLAGS.num_trashobjs,
    num_trashcans=FLAGS.num_trashcans,
    p_teleport=FLAGS.p_teleport,
    p_unsafe=FLAGS.p_unsafe,
    # lang wrapper config
    language_obs="token_embeds",  # TODO: later, make the obs handle arbitrary keys and make this a flag
    repeat_task_every=FLAGS.repeat_task_every,
    preread_max=FLAGS.preread_max,
    p_language=FLAGS.p_language,
    lang_types=['task'],
  )

  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if FLAGS.run_mode == 'actor':
    print('starting actor loop')
    actor.actor_loop(create_environment)
  elif FLAGS.run_mode == 'learner':
    print('starting learner loop')
    learner.learner_loop(create_environment,
                         create_agent,
                         create_optimizer)
  else:
    raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
  app.run(main)
