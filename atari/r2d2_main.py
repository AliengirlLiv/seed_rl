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


"""R2D2 binary for ATARI-57.

Actor and learner are in the same binary so that all flags are shared.
"""

import sys  # TODO: find another way to make this work!!
sys.path = ['/home/olivia/LangWorld/docker_seed', '/home/jessy/olivia/docker_seed', '/seed_rl'] + sys.path


from absl import app
from absl import flags
from seed_rl.agents.r2d2 import learner
from seed_rl.atari import env
from seed_rl.atari import networks
from seed_rl.common import actor
from seed_rl.common import common_flags  
import tensorflow as tf



FLAGS = flags.FLAGS

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.00048, 'Learning rate.')
flags.DEFINE_float('adam_epsilon', 1e-3, 'Adam epsilon.')

flags.DEFINE_integer('stack_size', 4, 'Number of frames to stack.')


def create_agent(env_observation_space, num_actions):
  return networks.DuelingLSTMDQNNet(
      num_actions, env_observation_space, FLAGS.stack_size, lang_key='none')


def create_optimizer(unused_final_iteration):
  learning_rate_fn = lambda iteration: FLAGS.learning_rate
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate,
                                       epsilon=FLAGS.adam_epsilon)
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
