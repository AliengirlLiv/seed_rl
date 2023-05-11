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


"""VTrace script for ATARI-57.
"""

import sys  # TODO: find another way to make this work!!
sys.path = ['/home/olivia/LangWorld/docker_seed', '/home/jessy/olivia/docker_seed', '/seed_rl'] + sys.path


from absl import app
from absl import flags
from seed_rl.agents.vtrace import learner
from seed_rl.atari import env
from seed_rl.agents.vtrace import networks
from seed_rl.common import actor
import tensorflow as tf


FLAGS = flags.FLAGS

# Optimizer settings.
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate.')
# Network settings.
flags.DEFINE_list('mlp_sizes', [64], 'Sizes of each of MLP hidden layer.')
flags.DEFINE_list('cnn_sizes', [16, 32, 32], 'Sizes of each of CNN hidden layer.')
flags.DEFINE_integer('lstm_size', 128, 'Size of the LSTM layer.')
flags.DEFINE_list('policy_sizes', [], 'Sizes of each of policy MLP hidden layer.')
flags.DEFINE_list('value_sizes', [], 'Sizes of each of value MLP hidden layer.')
# Environment settings.
flags.DEFINE_string('task_name', 's1', 'Messenger level (s1, s2, or s3)')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('length', 64, 'Length of environment.')
flags.DEFINE_integer('stack_size', 4, 'Number of frames to stack.')
flags.DEFINE_list('mlp_core_sizes', None, 'Sizes of each of value MLP (substitute for LSTM)')


def create_agent(action_space, env_observation_space,
                 unused_parametric_action_distribution):
    return networks.ImpalaDeep(action_space.n,
                             lstm_size=FLAGS.lstm_size,
                             mlp_sizes=[int(s) for s in FLAGS.mlp_sizes],
                             cnn_sizes=[int(s) for s in FLAGS.cnn_sizes],
                             vocab_size=-1,
                             lang_key='none',
                             policy_sizes=[int(size) for size in FLAGS.policy_sizes],
                             value_sizes=[int(size) for size in FLAGS.value_sizes],
                             obs_space=env_observation_space,
                             mlp_core_sizes=[int(size) for size in FLAGS.mlp_core_sizes] if FLAGS.mlp_core_sizes else None)


def create_optimizer(unused_final_iteration):
    # learning_rate_fn = lambda iteration: FLAGS.learning_rate
    #   optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(FLAGS.learning_rate, FLAGS.total_environment_frames, 0)
    optimizer = tf.keras.optimizers.RMSprop(FLAGS.learning_rate, momentum=0, epsilon=0.01)
    return optimizer, learning_rate_fn


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    create_env = lambda task, config: env.create_environment(task, config, dict_space=True)
    if FLAGS.run_mode == 'actor':
        actor.actor_loop(create_env)
    elif FLAGS.run_mode == 'learner':
        learner.learner_loop(create_env,
                            create_agent,
                            create_optimizer)
    else:
        raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
  app.run(main)
