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


"""R2D2 algo for Homecook

Actor and learner are in the same binary so that all flags are shared.
"""

import sys  # TODO: find another way to make this work!!
sys.path = ['/home/olivia/LangWorld/docker_seed', '/home/jessy/olivia/docker_seed', '/seed_rl'] + sys.path


from absl import app
from absl import flags
from seed_rl.agents.r2d2 import learner
from seed_rl.messenger_env import env
from seed_rl.atari import networks
from seed_rl.common import actor
import tensorflow as tf
import random



FLAGS = flags.FLAGS

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.00048, 'Learning rate.')
flags.DEFINE_float('adam_epsilon', 1e-3, 'Adam epsilon.')
# Network settings.
flags.DEFINE_list('mlp_sizes', [64], 'Sizes of each of MLP hidden layer.')
flags.DEFINE_list('cnn_sizes', [16, 32, 32], 'Sizes of each of CNN hidden layer.')
flags.DEFINE_list('cnn_strides', [4, 2, 1], 'Sizes of each of CNN hidden layer.')
flags.DEFINE_list('cnn_kernels', [8, 4, 3], 'Sizes of each of CNN hidden layer.')
flags.DEFINE_integer('lstm_size', 128, 'Size of the LSTM layer.')
flags.DEFINE_list('policy_sizes', None, 'Sizes of each of policy MLP hidden layer.')
flags.DEFINE_list('value_sizes', None, 'Sizes of each of value MLP hidden layer.')

flags.DEFINE_integer('stack_size', 1, 'Number of frames to stack.')
# Environment settings.
flags.DEFINE_string('task_name', 's1', 'Messenger level (s1, s2, or s3)')
flags.DEFINE_enum('lang_key', 'token', ['token', 'token_embed', 'sentence_embed', 'none'], 'Language key.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('length', 64, 'Length of environment.')
flags.DEFINE_list('aux_sizes', [256], 'Sizes of each of aux MLP hidden layer.')
flags.DEFINE_list('aux_heads', [], 'Auxiliary prediction heads.')
flags.DEFINE_list('mlp_core_sizes', None, 'Sizes of each of value MLP (substitute for LSTM)')



def create_agent(env_observation_space, num_actions):
    return networks.DuelingLSTMDQNNet(
        num_actions, env_observation_space,
        FLAGS.stack_size,
        lstm_size=FLAGS.lstm_size,
        mlp_sizes=[int(size) for size in FLAGS.mlp_sizes],
        cnn_sizes=[int(size) for size in FLAGS.cnn_sizes],
        cnn_strides=[int(stride) for stride in FLAGS.cnn_strides],
        cnn_kernels=[int(kernel) for kernel in FLAGS.cnn_kernels],
        vocab_size=env_observation_space[FLAGS.lang_key].high + 1 if FLAGS.lang_key == 'token' else env_observation_space[FLAGS.lang_key].shape[0],
        lang_key=FLAGS.lang_key,
        policy_sizes=[int(size) for size in FLAGS.policy_sizes] if FLAGS.policy_sizes else None,
        value_sizes=[int(size) for size in FLAGS.value_sizes] if FLAGS.value_sizes else None,
        aux_pred_sizes=[int(size) for size in FLAGS.aux_sizes],
        aux_pred_heads=FLAGS.aux_heads,
        mlp_core_sizes=[int(size) for size in FLAGS.mlp_core_sizes] if FLAGS.mlp_core_sizes else None,
        )


def create_optimizer(unused_final_iteration):
  learning_rate_fn = lambda iteration: FLAGS.learning_rate
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate,
                                       epsilon=FLAGS.adam_epsilon)
  return optimizer, learning_rate_fn


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    tf.random.set_seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    create_environment = lambda task, config: env.create_environment(
        task=FLAGS.task_name,
        mode='train',
        length=FLAGS.length,
        language_obs='token_embeds' if FLAGS.lang_key in ['token_embed', 'token'] else 'sentence_embeds',
    )
    if FLAGS.run_mode == 'actor':
        actor.actor_loop(create_environment)
    elif FLAGS.run_mode == 'learner':
        learner.learner_loop(create_environment,
                            create_agent,
                            create_optimizer)
    else:
        raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
  app.run(main)
