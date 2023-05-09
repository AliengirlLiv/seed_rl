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

"""MLP+LSTM network for use with V-trace."""

import collections
from seed_rl.common import utils
import tensorflow as tf
from seed_rl.dmlab.networks import _Stack

AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')


class MLPandLSTM(tf.Module):
  """MLP+stacked LSTM Agent."""

  def __init__(self, parametric_action_distribution, mlp_sizes, lstm_sizes):
    """Creates an MLP followed by a stacked LSTM agent.

    Args:
      parametric_action_distribution: an object of ParametricDistribution class
        specifing a parametric distribution over actions to be used
      mlp_sizes: list of integers with sizes of hidden MLP layers
      lstm_sizes: list of integers with sizes of LSTM layers
    """
    super(MLPandLSTM, self).__init__(name='MLPandLSTM')
    self._parametric_action_distribution = parametric_action_distribution

    # MLP
    mlp_layers = [tf.keras.layers.Dense(size, 'relu') for size in mlp_sizes]
    self._mlp = tf.keras.Sequential(mlp_layers)
    # stacked LSTM
    lstm_cells = [tf.keras.layers.LSTMCell(size) for size in lstm_sizes]
    self._core = tf.keras.layers.StackedRNNCells(lstm_cells)
    # Layers for _head.
    self._policy_logits = tf.keras.layers.Dense(
        parametric_action_distribution.param_size, name='policy_logits')
    self._baseline = tf.keras.layers.Dense(1, name='baseline')

  @tf.function
  def initial_state(self, batch_size):
    return self._core.get_initial_state(batch_size=batch_size, dtype=tf.float32)

  def _head(self, core_output):
    policy_logits = self._policy_logits(core_output)
    baseline = tf.squeeze(self._baseline(core_output), axis=-1)

    # Sample an action from the policy.
    action = self._parametric_action_distribution.sample(policy_logits)

    return AgentOutput(action, policy_logits, baseline)

  # Not clear why, but if "@tf.function" declarator is placed directly onto
  # __call__, training fails with "uninitialized variable *baseline".
  # when running on multiple learning tpu cores.


  @tf.function
  def get_action(self, *args, **kwargs):
    return self.__call__(*args, **kwargs)

  def __call__(self, prev_actions, env_outputs, core_state, unroll=False,
               is_training=False):
    """Runs the agent.

    Args:
      prev_actions: Previous action. Not used by this agent.
      env_outputs: Structure with reward, done and observation fields. Only
        observation field is used by this agent. It should have the shape
        [time, batch_size, observation_size].
      core_state: Agent state.
      unroll: Should be True if inputs contain the time dimension and False
        otherwise.
      is_training: Whether we are in the loss computation. Not used by this
        agent.
    Returns:
      A structure with action, policy_logits and baseline.
    """
    if not unroll:
      # Add time dimension.
      prev_actions, env_outputs = tf.nest.map_structure(
          lambda t: tf.expand_dims(t, 0), (prev_actions, env_outputs))

    outputs, core_state = self._unroll(prev_actions, env_outputs, core_state)

    if not unroll:
      # Remove time dimension.
      outputs = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

    return outputs, core_state

  def _unroll(self, unused_prev_actions, env_outputs, core_state):
    unused_reward, done, observation, _, _ = env_outputs
    observation = self._mlp(observation)

    initial_core_state = self._core.get_initial_state(
        batch_size=tf.shape(observation)[1], dtype=tf.float32)
    core_output_list = []
    for input_, d in zip(tf.unstack(observation), tf.unstack(done)):
      # If the episode ended, the core state should be reset before the next.
      core_state = tf.nest.map_structure(
          lambda x, y, d=d: tf.where(  
              tf.reshape(d, [d.shape[0]] + [1] * (x.shape.rank - 1)), x, y),
          initial_core_state,
          core_state)
      core_output, core_state = self._core(input_, core_state)
      core_output_list.append(core_output)
    outputs = tf.stack(core_output_list)

    return utils.batch_apply(self._head, (outputs,)), core_state


class ImpalaDeep(tf.Module):
  """Agent with ResNet.

  The deep model in
  "IMPALA: Scalable Distributed Deep-RL with
  Importance Weighted Actor-Learner Architectures"
  by Espeholt, Soyer, Munos et al.
  """

  def __init__(self, num_actions, mlp_sizes=(64,), cnn_sizes=(16, 32, 32), vocab_size=32100, lang_key='token', lstm_size=256, policy_sizes=(), value_sizes=()):
    super(ImpalaDeep, self).__init__(name='impala_deep')

    # Parameters and layers for unroll.
    self._num_actions = num_actions
    self._core = tf.keras.layers.LSTMCell(lstm_size)
    self._lang_key = lang_key
    self._vocab_size = vocab_size
    mlp_layers = []
    for i, size in enumerate(mlp_sizes):
      mlp_layers.append(tf.keras.layers.Dense(size, None))
      if i < len(mlp_sizes) - 1:
        mlp_layers.append(tf.keras.layers.Swish())
        mlp_layers.append(tf.keras.layers.LayerNormalization())
    self._mlp = tf.keras.Sequential(mlp_layers)

    # Parameters and layers for _torso.
    self._stacks = [
        _Stack(num_ch, 2)
        for num_ch in cnn_sizes
    ]
    self._conv_to_linear = tf.keras.layers.Dense(256)

    # Layers for _head.
    layer_list = []
    for size in policy_sizes:
      layer_list.append(tf.keras.layers.Dense(size, None))
      layer_list.append(tf.keras.layers.Swish())
      layer_list.append(tf.keras.layers.LayerNormalization())
    layer_list.append(tf.keras.layers.Dense(self._num_actions, name='policy_logits'))
    self._policy_logits = tf.keras.Sequential(layer_list)
    
    layer_list = []
    for size in value_sizes:
      layer_list.append(tf.keras.layers.Dense(size, None))
      layer_list.append(tf.keras.layers.Swish())
      layer_list.append(tf.keras.layers.LayerNormalization())
    layer_list.append(tf.keras.layers.Dense(1, name='baseline'))
    self._baseline = tf.keras.Sequential(layer_list)

  def initial_state(self, batch_size):
    return self._core.get_initial_state(batch_size=batch_size, dtype=tf.float32)

  def _torso(self, prev_action, env_output):
    reward, _, obs, _, _ = env_output
    frame = obs['image']

    # Convert to floats.
    frame = tf.cast(frame, tf.float32)

    frame /= 255
    conv_out = frame
    for stack in self._stacks:
      conv_out = stack(conv_out)

    conv_out = tf.nn.relu(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)

    conv_out = self._conv_to_linear(conv_out)
    conv_out = tf.nn.relu(conv_out)
    
    token = obs[self._lang_key]
    if self._lang_key == 'token':
      # One-hot encoding
      token = tf.cast(token, tf.int32)
      token = tf.one_hot(token, self._vocab_size)
    lang = self._mlp(token)
    
    # Append clipped last reward and one hot last action.
    clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
    one_hot_prev_action = tf.one_hot(prev_action, self._num_actions)
    return tf.concat([conv_out, lang, clipped_reward, one_hot_prev_action], axis=1)

  def _head(self, core_output):
    policy_logits = self._policy_logits(core_output)
    baseline = tf.squeeze(self._baseline(core_output), axis=-1)

    # Sample an action from the policy.
    new_action = tf.random.categorical(policy_logits, 1, dtype=tf.int64)
    new_action = tf.squeeze(new_action, 1, name='action')

    return AgentOutput(new_action, policy_logits, baseline)

  # Not clear why, but if "@tf.function" declarator is placed directly onto
  # __call__, training fails with "uninitialized variable *baseline".
  # when running on multiple learning tpu cores.


  @tf.function
  def get_action(self, *args, **kwargs):
    return self.__call__(*args, **kwargs)

  def __call__(self,
               prev_actions,
               env_outputs,
               core_state,
               unroll=False,
               is_training=False):
    
    reward, done, observation, abandoned, episode_step = env_outputs
    env_outputs = utils.EnvOutput(reward, done, observation, abandoned,
                                  episode_step)
        
    if not unroll:
      # Add time dimension.
      prev_actions, env_outputs = tf.nest.map_structure(
          lambda t: tf.expand_dims(t, 0), (prev_actions, env_outputs))
    outputs, core_state = self._unroll(prev_actions, env_outputs, core_state)
    if not unroll:
      # Remove time dimension.
      outputs = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)
        
    # Compute the number of parameters
    import numpy as np
    num_params = sum([np.prod(v.shape) for v in self.trainable_variables])

    # Print the number of parameters
    print("!"* 1000)
    print(f'Total number of parameters: {num_params}')
    if hasattr(self, '_embedding'):
      print(f'Number of embedding params: {sum([np.prod(v.shape) for v in self._embedding.trainable_variables])}')
    print(f'Number of lstm params: {sum([np.prod(v.shape) for v in self._core.trainable_variables])}')
    if hasattr(self, '_mlp'):
      print(f'Number of mlp params: {sum([np.prod(v.shape) for v in self._mlp.trainable_variables])}')
    print(f'Number of cnn params: {sum([sum([np.prod(v.shape) for v in s.trainable_variables]) for s in self._stacks])}')
    print(f'Number of conv_to_linear params: {sum([np.prod(v.shape) for v in self._conv_to_linear.trainable_variables])}')
    print(f'Number of policy_logits params: {sum([np.prod(v.shape) for v in self._policy_logits.trainable_variables])}')
    print(f'Number of baseline params: {sum([np.prod(v.shape) for v in self._baseline.trainable_variables])}')
    if hasattr(self, '_embedding'):
      print(f'Number of params not in embedding: {num_params - sum([np.prod(v.shape) for v in self._embedding.trainable_variables])}')

    return outputs, core_state

  def _unroll(self, prev_actions, env_outputs, core_state):
    unused_reward, done, unused_observation, _, _ = env_outputs

    torso_outputs = utils.batch_apply(self._torso, (prev_actions, env_outputs))

    initial_core_state = self._core.get_initial_state(
        batch_size=tf.shape(prev_actions)[1], dtype=tf.float32)
    core_output_list = []
    for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
      # If the episode ended, the core state should be reset before the next.
      core_state = tf.nest.map_structure(
          lambda x, y, d=d: tf.where(  
              tf.reshape(d, [d.shape[0]] + [1] * (x.shape.rank - 1)), x, y),
          initial_core_state,
          core_state)
      core_output, core_state = self._core(input_, core_state)
      core_output_list.append(core_output)
    core_outputs = tf.stack(core_output_list)

    return utils.batch_apply(self._head, (core_outputs,)), core_state
