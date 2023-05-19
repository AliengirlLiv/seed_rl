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

import sys  # TODO: find another way to make this work!!
sys.path = ['/home/olivia/LangWorld/docker_seed', '/home/jessy/olivia/docker_seed', '/seed_rl'] + sys.path


from absl import app
from absl import flags

import random
from seed_rl.agents.vtrace import learner
from seed_rl.agents.vtrace import networks
from seed_rl.common import actor
from seed_rl.vln import env
import tensorflow as tf


# Optimizer settings.
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate.')
# Network settings.
flags.DEFINE_list('mlp_sizes', [64], 'Sizes of each of MLP hidden layer.')
flags.DEFINE_list('cnn_sizes', [16, 32, 32], 'Sizes of each of CNN hidden layer.')
flags.DEFINE_integer('lstm_size', 128, 'Size of the LSTM layer.')
flags.DEFINE_list('policy_sizes', [], 'Sizes of each of policy MLP hidden layer.')
flags.DEFINE_list('value_sizes', [], 'Sizes of each of value MLP hidden layer.')
# Environment settings.
flags.DEFINE_string('dataset', 'train_parsed_ns_all_mips_all', 'Dataset.')
flags.DEFINE_integer('use_descriptions', 1, 'Use descriptions.')
flags.DEFINE_bool('use_depth', True, 'Use depth.')
flags.DEFINE_bool('use_stored_tokens', False, 'Use stored tokens.')
flags.DEFINE_integer('size', 64, 'Size.')
flags.DEFINE_string('mode', 'train', 'Mode.')
flags.DEFINE_float('use_expert', 0., 'Use expert.')
flags.DEFINE_float('min_use_expert', 0., 'Min use expert.')
flags.DEFINE_integer('anneal_expert_eps', 0, 'Anneal expert eps.')
flags.DEFINE_float('success_reward', 1000, 'Success reward.')
flags.DEFINE_float('early_stop_penalty', -10., 'Early stop penalty.')


FLAGS = flags.FLAGS


def create_agent(action_space, env_observation_space,
                 unused_parametric_action_distribution):
  return networks.ImpalaDeep(action_space.n,
                             lstm_size=FLAGS.lstm_size,
                             mlp_sizes=[int(size) for size in FLAGS.mlp_sizes],
                             cnn_sizes=[int(size) for size in FLAGS.cnn_sizes],
                             vocab_size=env_observation_space[FLAGS.lang_key].high + 1,
                             lang_key=FLAGS.lang_key,
                             policy_sizes=[int(size) for size in FLAGS.policy_sizes],
                             value_sizes=[int(size) for size in FLAGS.value_sizes],
                             obs_space=env_observation_space)


def create_optimizer(unused_final_iteration):
  learning_rate_fn = lambda iteration: FLAGS.learning_rate
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  return optimizer, learning_rate_fn


def main(argv):
  tf.random.set_seed(FLAGS.seed)
  random.seed(FLAGS.seed)
  def create_environment(task, config): return env.create_environment(
        dataset=FLAGS.dataset,
        language_obs='token_embeds' if FLAGS.lang_key in ['token_embed', 'token'] else 'sentence_embeds',
        use_descriptions=int(FLAGS.use_descriptions),
        mode=FLAGS.mode,
        use_depth=FLAGS.use_depth,
        use_stored_tokens=FLAGS.use_stored_tokens,
        size=(FLAGS.size, FLAGS.size),
        use_expert=FLAGS.use_expert,
        min_use_expert=FLAGS.min_use_expert,
        anneal_expert_eps=FLAGS.anneal_expert_eps,
        success_reward=FLAGS.success_reward,
        early_stop_penalty=FLAGS.early_stop_penalty,
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
