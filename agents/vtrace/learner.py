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

# python3
"""V-trace based SEED learner."""

import collections
import math
import os
import time

from absl import flags
from absl import logging

from seed_rl import grpc2
from seed_rl.common import common_flags  
from seed_rl.common import utils
from seed_rl.common import vtrace
from seed_rl.common.parametric_distribution import get_parametric_distribution_for_action_space

import tensorflow as tf


FLAGS = flags.FLAGS

# Training.
flags.DEFINE_integer('save_checkpoint_secs', 1800,
                     'Checkpoint save period in seconds.')
flags.DEFINE_integer('total_environment_frames', int(1e9),
                     'Total environment frames to train for.')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('inference_batch_size', -1,
                     'Batch size for inference, -1 for auto-tune.')
flags.DEFINE_integer('unroll_length', 100, 'Unroll length in agent steps.')
flags.DEFINE_integer('num_training_tpus', 1, 'Number of TPUs for training.')
flags.DEFINE_string('init_checkpoint', None,
                    'Path to the checkpoint used to initialize the agent.')

# Loss settings.
flags.DEFINE_float('entropy_cost', 0.00025, 'Entropy cost/multiplier.')
flags.DEFINE_float('target_entropy', None, 'If not None, the entropy cost is '
                   'automatically adjusted to reach the desired entropy level.')
flags.DEFINE_float('entropy_cost_adjustment_speed', 10., 'Controls how fast '
                   'the entropy cost coefficient is adjusted.')
flags.DEFINE_float('baseline_cost', .5, 'Baseline cost/multiplier.')
flags.DEFINE_float('kl_cost', 0., 'KL(old_policy|new_policy) loss multiplier.')
flags.DEFINE_float('discounting', .99, 'Discounting factor.')
flags.DEFINE_float('lambda_', 1., 'Lambda.')
flags.DEFINE_float('max_abs_reward', 0.,
                   'Maximum absolute reward when calculating loss.'
                   'Use 0. to disable clipping.')

# Logging
flags.DEFINE_integer('log_batch_frequency', 100, 'We average that many batches '
                     'before logging batch statistics like entropy.')
flags.DEFINE_integer('log_episode_frequency', 1, 'We average that many episodes'
                     ' before logging average episode return and length.')
flags.DEFINE_integer('use_wandb', 1, 'Whether to use wandb.')
flags.DEFINE_string('env', 'homecook', 'Environment.')  # TODO: move these all to one config file?
flags.DEFINE_string('exp_name', 'temp', 'Exp name, also used for wandb.')
flags.DEFINE_float('clip_norm', None, 'We clip gradient norm to this value.')

FLAGS = flags.FLAGS


def compute_loss(logger, parametric_action_distribution, agent, agent_state,
                 prev_actions, env_outputs, agent_outputs):
  learner_outputs, _ = agent(prev_actions,
                             env_outputs,
                             agent_state,
                             unroll=True,
                             is_training=True)

  # Use last baseline value (from the value function) to bootstrap.
  bootstrap_value = learner_outputs.baseline[-1]

  # At this point, we have unroll length + 1 steps. The last step is only used
  # as bootstrap value, so it's removed.
  agent_outputs = tf.nest.map_structure(lambda t: t[:-1], agent_outputs)
  rewards, done, _, _, _ = tf.nest.map_structure(lambda t: t[1:], env_outputs)
  learner_outputs = tf.nest.map_structure(lambda t: t[:-1], learner_outputs)

  if FLAGS.max_abs_reward:
    rewards = tf.clip_by_value(rewards, -FLAGS.max_abs_reward,
                               FLAGS.max_abs_reward)
  discounts = tf.cast(~done, tf.float32) * FLAGS.discounting

  target_action_log_probs = parametric_action_distribution.log_prob(
      learner_outputs.policy_logits, agent_outputs.action)
  behaviour_action_log_probs = parametric_action_distribution.log_prob(
      agent_outputs.policy_logits, agent_outputs.action)

  # Compute V-trace returns and weights.
  vtrace_returns = vtrace.from_importance_weights(
      target_action_log_probs=target_action_log_probs,
      behaviour_action_log_probs=behaviour_action_log_probs,
      discounts=discounts,
      rewards=rewards,
      values=learner_outputs.baseline,
      bootstrap_value=bootstrap_value,
      lambda_=FLAGS.lambda_)

  # Policy loss based on Policy Gradients
  policy_loss = -tf.reduce_mean(target_action_log_probs *
                                tf.stop_gradient(vtrace_returns.pg_advantages))

  # Value function loss
  v_error = vtrace_returns.vs - learner_outputs.baseline
  v_loss = FLAGS.baseline_cost * 0.5 * tf.reduce_mean(tf.square(v_error))

  # Entropy reward
  entropy = tf.reduce_mean(
      parametric_action_distribution.entropy(learner_outputs.policy_logits))
  entropy_loss = tf.stop_gradient(agent.entropy_cost()) * -entropy

  # KL(old_policy|new_policy) loss
  kl = behaviour_action_log_probs - target_action_log_probs
  kl_loss = FLAGS.kl_cost * tf.reduce_mean(kl)

  # Entropy cost adjustment (Langrange multiplier style)
  if FLAGS.target_entropy:
    entropy_adjustment_loss = agent.entropy_cost() * tf.stop_gradient(
        tf.reduce_mean(entropy) - FLAGS.target_entropy)
  else:
    entropy_adjustment_loss = 0. * agent.entropy_cost()  # to avoid None in grad

  total_loss = (policy_loss + v_loss + entropy_loss + kl_loss +
                entropy_adjustment_loss)

  # value function
  session = logger.log_session()
  logger.log(session, 'V/value function',
             tf.reduce_mean(learner_outputs.baseline))
  logger.log(session, 'V/L2 error', tf.sqrt(tf.reduce_mean(tf.square(v_error))))
  # losses
  logger.log(session, 'losses/policy', policy_loss)
  logger.log(session, 'losses/V', v_loss)
  logger.log(session, 'losses/entropy', entropy_loss)
  logger.log(session, 'losses/kl', kl_loss)
  logger.log(session, 'losses/total', total_loss)
  # policy
  dist = parametric_action_distribution.create_dist(
      learner_outputs.policy_logits)
  if hasattr(dist, 'scale'):
    logger.log(session, 'policy/std', tf.reduce_mean(dist.scale))
  logger.log(session, 'policy/max_action_abs(before_tanh)',
             tf.reduce_max(tf.abs(agent_outputs.action)))
  logger.log(session, 'policy/entropy', entropy)
  logger.log(session, 'policy/entropy_cost', agent.entropy_cost())
  logger.log(session, 'policy/kl(old|new)', tf.reduce_mean(kl))

  return total_loss, session


Unroll = collections.namedtuple(
    'Unroll', 'agent_state prev_actions env_outputs agent_outputs')


def validate_config():
  utils.validate_learner_config(FLAGS)


def learner_loop(create_env_fn, create_agent_fn, create_optimizer_fn):
  """Main learner loop.

  Args:
    create_env_fn: Callable that must return a newly created environment. The
      callable takes the task ID as argument - an arbitrary task ID of 0 will be
      passed by the learner. The returned environment should follow GYM's API.
      It is only used for infering tensor shapes. This environment will not be
      used to generate experience.
    create_agent_fn: Function that must create a new tf.Module with the neural
      network that outputs actions and new agent state given the environment
      observations and previous agent state. See dmlab.agents.ImpalaDeep for an
      example. The factory function takes as input the environment action and
      observation spaces and a parametric distribution over actions.
    create_optimizer_fn: Function that takes the final iteration as argument
      and must return a tf.keras.optimizers.Optimizer and a
      tf.keras.optimizers.schedules.LearningRateSchedule.
  """
  logging.info('Starting learner loop')
  validate_config()
  
  config = FLAGS
  if config.use_wandb == 1:
    import wandb
    import pathlib
    logdir = pathlib.Path(FLAGS.logdir) / FLAGS.exp_name

    wandb_id_file = f"{str(logdir)}/wandb_id.txt"
    wandb_pa = pathlib.Path(wandb_id_file)
    if wandb_pa.exists():
      print("!! Resuming wandb run !!")
      with open(wandb_id_file, "r") as f:
        wandb_id = f.read().strip()
    else:
      logdir.mkdir(parents=True, exist_ok=True)
      wandb_id = wandb.util.generate_id()
      with open(wandb_id_file, "w") as f:
        f.write(wandb_id)
    if "homegrid" in config.env:
      project = "homegridv3"
    elif "messenger" in config.env:
      project = "messenger"
    elif "homecook" in config.env:
      project = "homecook"
    elif "vln" in config.env:
      project = "vln"
    else:
      project = 'debug'
    wandb.init(
      id=wandb_id,
      resume="allow",
      project=project,
      name=config.exp_name,
      group=config.exp_name[:config.exp_name.rfind("_")],
      sync_tensorboard=True,
      config=config.flag_values_dict(),
    )
  
  settings = utils.init_learner_multi_host(FLAGS.num_training_tpus)
  strategy, hosts, training_strategy, encode, decode = settings
  env = create_env_fn(0, FLAGS)
  parametric_action_distribution = get_parametric_distribution_for_action_space(
      env.action_space)
  
  obs_shape = {k: tf.TensorSpec(env.observation_space[k].shape, env.observation_space[k].dtype, k) for k in env.observation_space}
  env_output_specs = utils.EnvOutput(
      tf.TensorSpec([], tf.float32, 'reward'),
      tf.TensorSpec([], tf.bool, 'done'),
      obs_shape,
      tf.TensorSpec([], tf.bool, 'abandoned'),
      tf.TensorSpec([], tf.int32, 'episode_step'),
  )
  action_specs = tf.TensorSpec(env.action_space.shape,
                               env.action_space.dtype, 'action')
  agent_input_specs = (action_specs, env_output_specs)

  # Initialize agent and variables.
  agent = create_agent_fn(env.action_space, env.observation_space,
                          parametric_action_distribution)
  initial_agent_state = agent.initial_state(1)
  agent_state_specs = tf.nest.map_structure(
      lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_state)
  unroll_specs = [None]  # Lazy initialization.
  input_ = tf.nest.map_structure(
      lambda s: tf.zeros([1] + list(s.shape), s.dtype), agent_input_specs)
  input_ = encode(input_)

  with strategy.scope():
    @tf.function
    def create_variables(*args):
      return agent.get_action(*decode(args))

    initial_agent_output, _ = create_variables(*input_, initial_agent_state)

    if not hasattr(agent, 'entropy_cost'):
      mul = FLAGS.entropy_cost_adjustment_speed
      agent.entropy_cost_param = tf.Variable(
          tf.math.log(FLAGS.entropy_cost) / mul,
          # Without the constraint, the param gradient may get rounded to 0
          # for very small values.
          constraint=lambda v: tf.clip_by_value(v, -20 / mul, 20 / mul),
          trainable=True,
          dtype=tf.float32)
      agent.entropy_cost = lambda: tf.exp(mul * agent.entropy_cost_param)
    # Create optimizer.
    iter_frame_ratio = (
        FLAGS.batch_size * FLAGS.unroll_length * FLAGS.num_action_repeats)
    final_iteration = int(
        math.ceil(FLAGS.total_environment_frames / iter_frame_ratio))
    optimizer, learning_rate_fn = create_optimizer_fn(final_iteration)


    iterations = optimizer.iterations
    optimizer._create_hypers()  
    optimizer._create_slots(agent.trainable_variables)  

    # ON_READ causes the replicated variable to act as independent variables for
    # each replica.
    temp_grads = [
        tf.Variable(tf.zeros_like(v), trainable=False,
                    synchronization=tf.VariableSynchronization.ON_READ)
        for v in agent.trainable_variables
    ]

  @tf.function
  def minimize(iterator):
    data = next(iterator)

    def compute_gradients(args):
      args = tf.nest.pack_sequence_as(unroll_specs[0], decode(args, data))
      with tf.GradientTape() as tape:
        loss, logs = compute_loss(logger, parametric_action_distribution, agent,
                                  *args)
      grads = tape.gradient(loss, agent.trainable_variables)
      gradient_norm_before_clip = tf.linalg.global_norm(grads)
      if FLAGS.clip_norm is not None:
        grads, _ = tf.clip_by_global_norm(
            grads, FLAGS.clip_norm, use_norm=gradient_norm_before_clip)
      for t, g in zip(temp_grads, grads):
        t.assign(g)
      return loss, logs

    loss, logs = training_strategy.run(compute_gradients, (data,))
    loss = training_strategy.experimental_local_results(loss)[0]

    def apply_gradients(_):
      optimizer.apply_gradients(zip(temp_grads, agent.trainable_variables))

    strategy.run(apply_gradients, (loss,))

    getattr(agent, 'end_of_training_step_callback',
            lambda: logging.info('end_of_training_step_callback not found'))()

    logger.step_end(logs, training_strategy, iter_frame_ratio)

  agent_output_specs = tf.nest.map_structure(
      lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_output)

  # Setup checkpointing and restore checkpoint.
  ckpt = tf.train.Checkpoint(agent=agent, optimizer=optimizer)
  if FLAGS.init_checkpoint is not None:
    # If the checkpoint name already contains ckpt-, we can just use it.
    if (FLAGS.init_checkpoint).split('/')[-1].startswith('ckpt-'):
      checkpoint_path = FLAGS.init_checkpoint
    else:
      # Find a file in the init_checkpoint directory which starts with 'ckpt-' and ends with '.index'.
      possible_files = []
      for f in tf.io.gfile.listdir(FLAGS.init_checkpoint):
        if f.startswith('ckpt-') and f.endswith('.index'):
          possible_files.append(os.path.join(FLAGS.init_checkpoint, f)[:-6])
      if len(possible_files) == 0:
        raise ValueError('No checkpoint file found in %s' %
                        FLAGS.init_checkpoint)
      if len(possible_files) > 1:
        # Choose the most recent checkpoint.
        checkpoint_path = max([p + '.index' for p in possible_files], key=os.path.getctime)
        tf.print('Found multiple checkpoint files, choosing %s' %
                  checkpoint_path)
      checkpoint_path = possible_files[0]
    tf.print('Loading initial checkpoint from %s...' % checkpoint_path)
    ckpt.restore(checkpoint_path).assert_consumed()
    # If we're restoring from checkpoint, we need to reset the optimizer's step
    optimizer.iterations.assign(0)
    
  manager = tf.train.CheckpointManager(
      ckpt, FLAGS.logdir, max_to_keep=1, keep_checkpoint_every_n_hours=1)
  last_ckpt_time = 0  # Force checkpointing of the initial model.
  if manager.latest_checkpoint:
    logging.info('Restoring checkpoint: %s', manager.latest_checkpoint)
    ckpt.restore(manager.latest_checkpoint).assert_consumed()
    last_ckpt_time = time.time()

  # Logging.
  summary_writer = tf.summary.create_file_writer(
      FLAGS.logdir, flush_millis=20000, max_queue=1000)
  logger = utils.ProgressLogger(summary_writer=summary_writer,
                                starting_step=iterations * iter_frame_ratio)

  servers = []
  unroll_queues = []
  info_specs = (
      tf.TensorSpec([], tf.int64, 'episode_num_frames'),
      tf.TensorSpec([], tf.float32, 'episode_returns'),
      tf.TensorSpec([], tf.float32, 'episode_raw_returns'),
      tf.TensorSpec([], tf.int64, 'total_non_reading_frames'),
      tf.TensorSpec([], tf.int64, 'total_frames'),
  )

  info_queue = utils.StructuredFIFOQueue(-1, info_specs)

  def create_host(i, host, inference_devices):
    total_frames = tf.Variable(0, dtype=tf.int64)
    total_non_reading_frames = tf.Variable(0, dtype=tf.int64)
    with tf.device(host):
      server = grpc2.Server([FLAGS.server_address])

      store = utils.UnrollStore(
          FLAGS.num_envs, FLAGS.unroll_length,
          (action_specs, env_output_specs, agent_output_specs))
      env_run_ids = utils.Aggregator(FLAGS.num_envs,
                                     tf.TensorSpec([], tf.int64, 'run_ids'))
      env_infos = utils.Aggregator(FLAGS.num_envs, info_specs,
                                   'env_infos')

      # First agent state in an unroll.
      first_agent_states = utils.Aggregator(
          FLAGS.num_envs, agent_state_specs, 'first_agent_states')

      # Current agent state and action.
      agent_states = utils.Aggregator(
          FLAGS.num_envs, agent_state_specs, 'agent_states')
      actions = utils.Aggregator(FLAGS.num_envs, action_specs, 'actions')

      unroll_specs[0] = Unroll(agent_state_specs, *store.unroll_specs)
      unroll_queue = utils.StructuredFIFOQueue(1, unroll_specs[0])

      def add_batch_size(ts):
        return tf.TensorSpec([FLAGS.inference_batch_size] + list(ts.shape),
                             ts.dtype, ts.name)

      inference_specs = (
          tf.TensorSpec([], tf.int32, 'env_id'),
          tf.TensorSpec([], tf.int64, 'run_id'),
          env_output_specs,
          tf.TensorSpec([], tf.float32, 'raw_reward'),
      )
      inference_specs = tf.nest.map_structure(add_batch_size, inference_specs)
      def create_inference_fn(inference_device):
        @tf.function(input_signature=inference_specs)
        def inference(env_ids, run_ids, env_outputs, raw_rewards):
          # Reset the environments that had their first run or crashed.
          previous_run_ids = env_run_ids.read(env_ids)
          env_run_ids.replace(env_ids, run_ids)
          reset_indices = tf.where(
              tf.not_equal(previous_run_ids, run_ids))[:, 0]
          envs_needing_reset = tf.gather(env_ids, reset_indices)
          if tf.not_equal(tf.shape(envs_needing_reset)[0], 0):
            tf.print('Environment ids needing reset:', envs_needing_reset)
          env_infos.reset(envs_needing_reset)
          total_frames.assign_add(FLAGS.inference_batch_size)
          reading = env_outputs.observation.get('is_read_step', tf.zeros_like(env_outputs.reward, dtype=tf.bool))
          total_non_reading_frames.assign_add(tf.reduce_sum(1 - tf.cast(reading, tf.int64)))
          store.reset(envs_needing_reset)
          initial_agent_states = agent.initial_state(
              tf.shape(envs_needing_reset)[0])
          first_agent_states.replace(envs_needing_reset, initial_agent_states)
          agent_states.replace(envs_needing_reset, initial_agent_states)
          actions.reset(envs_needing_reset)

          tf.debugging.assert_non_positive(
              tf.cast(env_outputs.abandoned, tf.int32),
              'Abandoned done states are not supported in VTRACE.')

          # Update steps and return.
          env_infos.add(env_ids, (0, env_outputs.reward, raw_rewards, 0, 0))
          done_ids = tf.gather(env_ids, tf.where(env_outputs.done)[:, 0])
          if i == 0:
            info_queue.enqueue_many(env_infos.read(done_ids))
          env_infos.reset(done_ids)
          env_infos.add(env_ids, (FLAGS.num_action_repeats, 0., 0., total_non_reading_frames * tf.cast(env_outputs.done, tf.int64), total_frames * tf.cast(env_outputs.done, tf.int64)))

          # Inference.
          prev_actions = actions.read(env_ids)
          input_ = encode((prev_actions, env_outputs))
          prev_agent_states = agent_states.read(env_ids)
          with tf.device(inference_device):
            @tf.function
            def agent_inference(*args):
              return agent(*decode(args), is_training=False)

            agent_outputs, curr_agent_states = agent_inference(
                *input_, prev_agent_states)

          # Append the latest outputs to the unroll and insert completed unrolls
          # in queue.
          completed_ids, unrolls = store.append(
              env_ids, (prev_actions, env_outputs, agent_outputs))
          unrolls = Unroll(first_agent_states.read(completed_ids), *unrolls)
          unroll_queue.enqueue_many(unrolls)
          first_agent_states.replace(completed_ids,
                                     agent_states.read(completed_ids))

          # Update current state.
          agent_states.replace(env_ids, curr_agent_states)
          actions.replace(env_ids, agent_outputs.action)
          # Return environment actions to environments.
          return agent_outputs.action

        return inference

      with strategy.scope():
        server.bind([create_inference_fn(d) for d in inference_devices])
      server.start()
      unroll_queues.append(unroll_queue)
      servers.append(server)

  for i, (host, inference_devices) in enumerate(hosts):
    print('Number of hosts: %d' % len(hosts))
    create_host(i, host, inference_devices)

  def dequeue(ctx):
    # Create batch (time major).
    env_outputs = tf.nest.map_structure(lambda *args: tf.stack(args), *[
        unroll_queues[ctx.input_pipeline_id].dequeue()
        for i in range(ctx.get_per_replica_batch_size(FLAGS.batch_size))
    ])
    env_outputs = env_outputs._replace(
        prev_actions=utils.make_time_major(env_outputs.prev_actions),
        env_outputs=utils.make_time_major(env_outputs.env_outputs),
        agent_outputs=utils.make_time_major(env_outputs.agent_outputs))
    env_outputs = env_outputs._replace(
        env_outputs=encode(env_outputs.env_outputs))
    # tf.data.Dataset treats list leafs as tensors, so we need to flatten and
    # repack.
    return tf.nest.flatten(env_outputs)

  def dataset_fn(ctx):
    dataset = tf.data.Dataset.from_tensors(0).repeat(None)

    def _dequeue(_):
      return dequeue(ctx)

    return dataset.map(
        _dequeue, num_parallel_calls=ctx.num_replicas_in_sync // len(hosts))

  dataset = training_strategy.experimental_distribute_datasets_from_function(
      dataset_fn)
  it = iter(dataset)

  def additional_logs():
    tf.summary.scalar('learning_rate', learning_rate_fn(iterations))
    n_episodes = info_queue.size()
    n_episodes -= n_episodes % FLAGS.log_episode_frequency
    if tf.not_equal(n_episodes, 0):
      episode_stats = info_queue.dequeue_many(n_episodes)
      episode_keys = [
          'episode_num_frames', 'episode_return', 'episode_raw_return'
      ]
      for key, values in zip(episode_keys, episode_stats[:len(episode_keys)]):
        for value in tf.split(values,
                              values.shape[0] // FLAGS.log_episode_frequency):
          tf.summary.scalar(key, tf.reduce_mean(value))
      global_keys = [
          'total_non_reading_frames', 'total_frames'
      ]
      for key, values in zip(global_keys, episode_stats[len(episode_keys):]):
        for value in tf.split(values,
                              values.shape[0] // FLAGS.log_episode_frequency):
          tf.summary.scalar(key, tf.reduce_max(value))

      for (frames, ep_return, raw_return, total_non_reading, total_frames) in zip(*episode_stats):
        logging.info('Server %i; Return: %f Raw return: %f Frames: %i, Non reading: %i, Total frames: %i', i, ep_return,
                     raw_return, frames, total_non_reading, total_frames)

  logger.start(additional_logs)
  # Execute learning.
  while iterations < final_iteration:
    print('ITERATIONS', iterations)
    # Save checkpoint.
    current_time = time.time()
    if current_time - last_ckpt_time >= FLAGS.save_checkpoint_secs:
      manager.save()
      # Apart from checkpointing, we also save the full model (including
      # the graph). This way we can load it after the code/parameters changed.
      tf.saved_model.save(agent, os.path.join(FLAGS.logdir, 'saved_model'))
      last_ckpt_time = current_time
    minimize(it)
  logger.shutdown()
  manager.save()
  tf.saved_model.save(agent, os.path.join(FLAGS.logdir, 'saved_model'))
  for server in servers:
    server.shutdown()
  for unroll_queue in unroll_queues:
    unroll_queue.close()
