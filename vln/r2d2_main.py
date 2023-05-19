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
from seed_rl.vln import env
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
# Environment settings
flags.DEFINE_list('lang_types', ['task'], 'Language types.')
flags.DEFINE_enum('lang_key', 'token', ['token', 'token_embed', 'sentence_embed', 'none'], 'Language key.')
flags.DEFINE_integer('seed', 0, 'Random seed.')

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



def create_agent(env_observation_space, num_actions):
    return networks.DuelingLSTMDQNNet(
        num_actions, env_observation_space,
        FLAGS.stack_size,
        lstm_size=FLAGS.lstm_size,
        mlp_sizes=[int(size) for size in FLAGS.mlp_sizes],
        cnn_sizes=[int(size) for size in FLAGS.cnn_sizes],
        cnn_strides=[int(stride) for stride in FLAGS.cnn_strides],
        cnn_kernels=[int(kernel) for kernel in FLAGS.cnn_kernels],
        vocab_size=env_observation_space[FLAGS.lang_key].high + 1,
        lang_key=FLAGS.lang_key,
        policy_sizes=[int(size) for size in FLAGS.policy_sizes] if FLAGS.policy_sizes else None,
        value_sizes=[int(size) for size in FLAGS.value_sizes] if FLAGS.value_sizes else None,
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
