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

import sys
sys.path = ['/home/jessy/olivia/docker_seed', '/seed_rl'] + sys.path


from absl import app
from absl import flags


import os
from seed_rl.agents.vtrace import learner
# from seed_rl.agents.vtrace import networks
from seed_rl.dmlab import networks
from seed_rl.common import actor
from seed_rl.common import common_flags  
from seed_rl.common import normalizer
from seed_rl.messenger_env import env
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
flags.DEFINE_string('task_name', 's1', 'Messenger level (s1, s2, or s3)')
flags.DEFINE_bool('separate_sentences', True, 'Split sentences in encoding.')

FLAGS = flags.FLAGS


# def create_agent(unused_action_space, unused_env_observation_space,
#                  parametric_action_distribution):
#   policy = networks.MLPandLSTM(
#       parametric_action_distribution,
#       mlp_sizes=[FLAGS.mlp_size] * FLAGS.n_mlp_layers,
#       lstm_sizes=[FLAGS.lstm_size] * FLAGS.n_lstm_layers)
#   if FLAGS.normalize_observations:
#     policy = normalizer.NormalizeObservationsWrapper(policy,
#                                                      normalizer.Normalizer())
#   return policy

def create_agent(action_space, unused_env_observation_space,
                 unused_parametric_action_distribution):
  return networks.ImpalaDeep(action_space.n)


def create_optimizer(unused_final_iteration):
  learning_rate_fn = lambda iteration: FLAGS.learning_rate
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  return optimizer, learning_rate_fn


def main(argv):
  create_environment = lambda task, config: env.create_environment(
    task=FLAGS.task_name,
    mode='train',
    separate_sentences=FLAGS.separate_sentences,
    message_prob=.2,
  )

  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if FLAGS.run_mode == 'actor':
    print('starting actor loop')
    actor.actor_loop(create_environment)
    print('ending actor loop')
  elif FLAGS.run_mode == 'learner':
    print('starting learner loop')
    learner.learner_loop(create_environment,
                         create_agent,
                         create_optimizer)
    print('ending learner loop')
  else:
    raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
  print("GOT TO MAIN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  app.run(main)
