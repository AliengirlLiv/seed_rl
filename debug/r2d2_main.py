import sys  # TODO: find another way to make this work!!
sys.path = ['/home/olivia/LangWorld/docker_seed', '/home/jessy/olivia/docker_seed', '/seed_rl'] + sys.path


from absl import app
from absl import flags
from seed_rl.agents.r2d2 import learner
from seed_rl.debug import env
from seed_rl.atari import networks
from seed_rl.common import actor
import tensorflow as tf
import random



FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.00048, 'Learning rate.')
flags.DEFINE_float('adam_epsilon', 1e-3, 'Adam epsilon.')
flags.DEFINE_list('policy_sizes', None, 'Sizes of each of policy MLP hidden layer.')
flags.DEFINE_list('value_sizes', None, 'Sizes of each of value MLP hidden layer.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_list('cnn_kernels', [8, 4, 3], 'Sizes of each of CNN hidden layer.')
flags.DEFINE_list('cnn_strides', [4, 2, 1], 'Sizes of each of CNN hidden layer.')

flags.DEFINE_integer('stack_size', 1, 'Number of frames to stack.')
flags.DEFINE_enum('lang_key', 'token_embed', ['token', 'token_embed', 'sentence_embed', 'none'], 'Language key.')
flags.DEFINE_list('aux_sizes', [256], 'Sizes of each of aux MLP hidden layer.')
flags.DEFINE_list('aux_heads', [], 'Auxiliary prediction heads.')
flags.DEFINE_enum('obs_mode', 'random', ['random', 'deterministic', 'action'], 'Obs mode')
flags.DEFINE_enum('done_mode', 'random', ['random', 'deterministic', 'action'], 'Done mode')
flags.DEFINE_enum('reward_mode', 'random', ['random', 'deterministic', 'action'], 'Reward mode')



def create_agent(env_observation_space, num_actions):
    return networks.DuelingLSTMDQNNet(
        num_actions, env_observation_space,
        FLAGS.stack_size,
        cnn_strides=[int(stride) for stride in FLAGS.cnn_strides],
        cnn_kernels=[int(kernel) for kernel in FLAGS.cnn_kernels],
        vocab_size=env_observation_space[FLAGS.lang_key].shape[0],
        lang_key=FLAGS.lang_key,
        aux_pred_sizes=[int(size) for size in FLAGS.aux_sizes],
        aux_pred_heads=FLAGS.aux_heads,
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
        obs_mode=FLAGS.obs_mode,
        done_mode=FLAGS.done_mode,
        reward_mode=FLAGS.reward_mode,
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
