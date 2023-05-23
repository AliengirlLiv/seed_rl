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
"""SEED agent using Keras."""

import collections
from seed_rl.common import utils
import tensorflow as tf
import numpy as np

AgentState = collections.namedtuple(
    # core_state: the opaque state of the recurrent core of the agent.
    # frame_stacking_state: a list of the last (stack_size - 1) observations
    #   with shapes typically [batch_size, height, width, 1].
    'AgentState', 'core_state frame_stacking_state')

STACKING_STATE_DTYPE = tf.int32


def initial_frame_stacking_state(stack_size, batch_size, observation_shape):
  """Returns the initial frame stacking state.

  It should match what stack_frames accepts and produces.

  Args:
    stack_size: int, the number of frames that should be stacked to form the
      observation provided to the neural network. stack_size=1 corresponds to no
      stacking.
    batch_size: int tensor.
    observation_shape: list, shape of a single observation, e.g.
      [height, width, 1].

  Returns:
    <STACKING_STATE_DTYPE>[batch_size, prod(observation_shape)] or an empty
    tuple if stack_size=1.
  """
  if stack_size == 1:
    return ()
  return tf.zeros(
      tf.concat([[batch_size], [tf.math.reduce_prod(observation_shape)]],
                axis=0),
      dtype=STACKING_STATE_DTYPE)


def stack_frames(frames, frame_stacking_state, done, stack_size):
  """Stacks frames.

  The [height, width] center dimensions of the tensors below are typical, but
  could be anything else consistent between tensors.

  Args:
    frames: <float32>[time, batch_size, height, width, channels]. These should
      be un-normalized frames in range [0, 255]. channels must be equal to 1
      when we actually stack frames (stack_size > 1).
    frame_stacking_state: If stack_size > 1, <int32>[batch_size, height*width].
      () if stack_size=1.
      Frame are bit-packed. The LSBs correspond to the oldest frames, MSBs to
      newest. Frame stacking state contains un-normalized frames in range
      [0, 256). We use [height*width] for the observation shape instead of
      [height, width] because it speeds up transfers to/from TPU by a factor ~2.
    done: <bool>[time, batch_size]
    stack_size: int, the number of frames to stack.
  Returns:
    A pair:
      - stacked frames, <float32>[time, batch_size, height, width, stack_size]
        tensor (range [0, 255]). Along the stack dimensions, frames go from
        newest to oldest.
      - New stacking state with the last (stack_size-1) frames.
  """
  if frames.shape[0:2] != done.shape[0:2]:
    raise ValueError(
        'Expected same first 2 dims for frames and dones. Got {} vs {}.'.format(
            frames.shape[0:2], done.shape[0:2]))
  batch_size = frames.shape[1]
  obs_shape = frames.shape[2:-1]
  if stack_size > 4:
    raise ValueError('Only up to stack size 4 is supported due to bit-packing.')
  if stack_size > 1 and frames.shape[-1] != 1:
      print('STACK SIZE', stack_size)
      print('FRAMES', frames.shape)
      raise ValueError('Due to frame stacking, we require last observation '
                     'dimension to be 1. Got {}'.format(frames.shape[-1]))
  if stack_size == 1:
    return frames, ()
  if frame_stacking_state[0].dtype != STACKING_STATE_DTYPE:
    raise ValueError('Expected dtype {} got {}'.format(
        STACKING_STATE_DTYPE, frame_stacking_state[0].dtype))

  frame_stacking_state = tf.reshape(
      frame_stacking_state, [batch_size] + obs_shape)

  # Unpacked 'frame_stacking_state'. Ordered from oldest to most recent.
  unstacked_state = []
  for i in range(stack_size - 1):
    # [batch_size, height, width]
    unstacked_state.append(tf.cast(tf.bitwise.bitwise_and(
        tf.bitwise.right_shift(frame_stacking_state, i * 8), 0xFF),
                                   tf.float32))

  # Same as 'frames', but with the previous (stack_size - 1) frames from
  # frame_stacking_state prepended.
  # [time+stack_size-1, batch_size, height, width, 1]
  extended_frames = tf.concat(
      [tf.reshape(frame, [1] + frame.shape + [1])
       for frame in unstacked_state] +
      [frames],
      axis=0)

  # [time, batch_size, height, width, stack_size].
  # Stacked frames, but does not take 'done' into account. We need to zero-out
  # the frames that cross episode boundaries.
  # Along the stack dimensions, frames go from newest to oldest.
  stacked_frames = tf.concat(
      [extended_frames[stack_size - 1 - i:extended_frames.shape[0] - i]
       for i in range(stack_size)],
      axis=-1)

  # We create a mask that contains true when the frame should be zeroed out.
  # Setting the final shape of the mask early actually makes computing
  # stacked_done_masks a few times faster.
  done_mask_row_shape = done.shape[0:2] + [1] * (frames.shape.rank - 2)
  done_masks = [
      tf.zeros(done_mask_row_shape, dtype=tf.bool),
      tf.reshape(done, done_mask_row_shape)
  ]
  while len(done_masks) < stack_size:
    previous_row = done_masks[-1]
    # Add 1 zero in front (time dimension).
    done_masks.append(
        tf.math.logical_or(
            previous_row,
            tf.pad(previous_row[:-1],
                   [[1, 0]] + [[0, 0]] * (previous_row.shape.rank - 1))))

  # This contains true when the frame crosses an episode boundary and should
  # therefore be zeroed out.
  # Example: ignoring batch_size, if done is [0, 1, 0, 0, 1, 0], stack_size=4,
  # this will be:
  # [[0 0, 0, 0, 0, 0],
  #  [0 1, 0, 0, 1, 0],
  #  [0 1, 1, 0, 1, 1],
  #  [0 1, 1, 1, 1, 1]].T
  # <bool>[time, batch_size, 1, 1, stack_size].
  stacked_done_masks = tf.concat(done_masks, axis=-1)
  stacked_frames = tf.where(
      stacked_done_masks,
      tf.zeros_like(stacked_frames), stacked_frames)

  # Build the new bit-packed state.
  # We construct the new state from 'stacked_frames', to make sure frames
  # before done is true are zeroed out.
  # This shifts the stack_size-1 items of the last dimension of
  # 'stacked_frames[-1, ..., :-1]'.
  shifted = tf.bitwise.left_shift(
      tf.cast(stacked_frames[-1, ..., :-1], tf.int32),
      # We want to shift so that MSBs are newest frames.
      [8 * i for i in range(stack_size - 2, -1, -1)])
  # This is really a reduce_or, because bits don't overlap.
  new_state = tf.reduce_sum(shifted, axis=-1)

  new_state = tf.reshape(new_state, [batch_size, obs_shape.num_elements()])

  return stacked_frames, new_state


def _unroll_cell(inputs, done, start_state, zero_state, recurrent_cell):
  """Applies a recurrent cell on inputs, taking care of managing state.

  Args:
    inputs: A tensor of shape [time, batch_size, <remaining dims>]. These are
      the inputs passed to the recurrent cell.
    done: <bool>[time, batch_size].
    start_state: Recurrent cell state at the beginning of the input sequence.
      Opaque tf.nest structure of tensors with batch front dimension.
    zero_state: Blank recurrent cell state. The current recurrent state will be
      replaced by this blank state whenever 'done' is true. Same shape as
      'start_state'.
    recurrent_cell: Function that will be applied at each time-step. Takes
      (input_t: [batch_size, <remaining dims>], current_state) as input, and
      returns (output_t: [<cell output dims>], new_state).

  Returns:
    A pair:
      - The time-stacked outputs of the recurrent cell. Shape [time,
        <cell output dims>].
      - The last state output by the recurrent cell.
  """
  stacked_outputs = []
  state = start_state
  inputs_list = tf.unstack(inputs)
  done_list = tf.unstack(done)
  assert len(inputs_list) == len(done_list), (
      "Inputs and done tensors don't have same time dim {} vs {}".format(
          len(inputs_list), len(done_list)))
  # Loop over time dimension.
  # input_t: [batch_size, batch_size, <remaining dims>].
  # done_t: [batch_size].
  for input_t, done_t in zip(inputs_list, done_list):
    # If the episode ended, the frame state should be reset before the next.
    state = tf.nest.map_structure(
        lambda x, y, done_t=done_t: tf.where(  
            tf.reshape(done_t, [done_t.shape[0]] + [1] *
                       (x.shape.rank - 1)), x, y),
        zero_state,
        state)
    output_t, state = recurrent_cell(input_t, state)
    stacked_outputs.append(output_t)
  return tf.stack(stacked_outputs), state


class DuelingLSTMDQNNet(tf.Module):
  """The recurrent network used to compute the agent's Q values.

  This is the dueling LSTM net similar to the one described in
  https://openreview.net/pdf?id=rkHVZWZAZ (only the Q(s, a) part), with the
  layer sizes mentioned in the R2D2 paper
  (https://openreview.net/pdf?id=r1lyTjAqYX), section Hyper parameters.
  """

  def __init__(self, num_actions, observation_space, stack_size=1, lang_key='token',
               mlp_sizes=(64,), cnn_sizes=(16, 32, 32), cnn_strides=(4, 2, 1), cnn_kernels=(8, 4, 3), 
               vocab_size=32100, policy_sizes=None, value_sizes=None, lstm_size=256, mlp_core_sizes=None,
               aux_pred_sizes=(256,),
              #  aux_pred_heads=('reward', 'done', 'lang', 'next_lang', 'image', 'next_image')
               aux_pred_heads=(),
               ):
    super(DuelingLSTMDQNNet, self).__init__(name='dueling_lstm_dqn_net')
    self._num_actions = num_actions
    self._use_lstm = lstm_size > 0
    self._uses_int_input = (observation_space['image'].high == 255).all()
    self._aux_pred_heads = aux_pred_heads
    agent_output_names = ['action', 'q_values'] + list(aux_pred_heads)
    self.AgentOutput = collections.namedtuple('AgentOutput', ' '.join(agent_output_names))
    layer_list = []
    assert len(cnn_sizes) == len(cnn_strides) == len(cnn_kernels)
    for size, stride, kernel in zip(cnn_sizes, cnn_strides, cnn_kernels):
      layer_list.append(tf.keras.layers.Conv2D(size, [kernel, kernel], stride,
                                               padding='valid', activation='relu'))
    layer_list.append(tf.keras.layers.Flatten())
    layer_list.append(tf.keras.layers.Dense(512, activation='relu'))
    
    self._body = tf.keras.Sequential(layer_list)
    if value_sizes is None:
      self._value = tf.keras.Sequential([
          tf.keras.layers.Dense(512, activation='relu', name='hidden_value'),
          tf.keras.layers.Dense(1, name='value_head'),
      ])
    else:
      layer_list = []
      for size in value_sizes:
        layer_list.append(tf.keras.layers.Dense(size, None))
        layer_list.append(tf.keras.layers.Activation(tf.keras.activations.swish))
        layer_list.append(tf.keras.layers.LayerNormalization())
      layer_list.append(tf.keras.layers.Dense(1, name='baseline'))
      self._value = tf.keras.Sequential(layer_list)
    
    if policy_sizes is None:
      self._advantage = tf.keras.Sequential([
          tf.keras.layers.Dense(512, activation='relu', name='hidden_advantage'),
          tf.keras.layers.Dense(self._num_actions, use_bias=False,
                                name='advantage_head'),
      ])
    else:
      layer_list = []
      for size in policy_sizes:
        layer_list.append(tf.keras.layers.Dense(size, None))
        layer_list.append(tf.keras.layers.Activation(tf.keras.activations.swish))
        layer_list.append(tf.keras.layers.LayerNormalization())
      layer_list.append(tf.keras.layers.Dense(self._num_actions, name='advantage_head', use_bias=False))
      self._advantage = tf.keras.Sequential(layer_list)
    
    if self._use_lstm:
      self._core = tf.keras.layers.LSTMCell(lstm_size)
    else:
      core_layers = []
      for i, size in enumerate(mlp_core_sizes):
        core_layers.append(tf.keras.layers.Dense(size, None))
        if i < len(mlp_core_sizes) - 1:
          core_layers.append(tf.keras.layers.Activation(tf.keras.activations.swish))
          core_layers.append(tf.keras.layers.LayerNormalization())
      self._core = tf.keras.Sequential(core_layers)
    self._lang_key = lang_key
    self._vocab_size = vocab_size
    if not lang_key == 'none':
      mlp_layers = []
      for i, size in enumerate(mlp_sizes):
        mlp_layers.append(tf.keras.layers.Dense(size, None))
        if i < len(mlp_sizes) - 1:
          mlp_layers.append(tf.keras.layers.Activation(tf.keras.activations.swish))
          mlp_layers.append(tf.keras.layers.LayerNormalization())
      self._mlp = tf.keras.Sequential(mlp_layers)

    if len(aux_pred_heads) > 0:
      layers = []
      for i, size in enumerate(aux_pred_sizes):
        layers.append(tf.keras.layers.Dense(size, None))
        layers.append(tf.keras.layers.Activation(tf.keras.activations.swish))
        layers.append(tf.keras.layers.LayerNormalization())
      self._aux_trunk = tf.keras.Sequential(layers)

    for pred_head in aux_pred_heads:
      if 'image' in pred_head: # Deconv
        if pred_head in ['image', 'next_image']:
          final_size = 3
        elif pred_head in ['lang', 'next_lang']:
          final_size = vocab_size
        else:
          raise ValueError('Unknown aux pred head: {}'.format(pred_head))
        deconv_layers = []
        for i, size in enumerate(aux_pred_sizes):
          deconv_layers.append(tf.keras.layers.Dense(size, None))
          deconv_layers.append(tf.keras.layers.Activation(tf.keras.activations.swish))
          deconv_layers.append(tf.keras.layers.LayerNormalization())
        deconv_layers.append(tf.keras.layers.Dense(np.prod(observation_space['image'].shape), name=pred_head))
        deconv_layers.append(tf.keras.layers.Reshape(observation_space['image'].shape))
        self.__setattr__('_pred_{}'.format(pred_head), tf.keras.Sequential(deconv_layers))
      else:  # MLP
        if pred_head in ['reward', 'done']:
          final_size = 1
        elif pred_head in ['lang', 'next_lang']:
          final_size = vocab_size
        else:
          raise ValueError('Unknown aux pred head: {}'.format(pred_head))
        mlp_layers = []
        for i, size in enumerate(aux_pred_sizes):
          mlp_layers.append(tf.keras.layers.Dense(size, None))
          mlp_layers.append(tf.keras.layers.Activation(tf.keras.activations.swish))
          mlp_layers.append(tf.keras.layers.LayerNormalization())
        mlp_layers.append(tf.keras.layers.Dense(final_size, name=pred_head))
        self.__setattr__('_pred_{}'.format(pred_head), tf.keras.Sequential(mlp_layers))

    self._observation_space = observation_space
    self._stack_size = stack_size

  def initial_state(self, batch_size):
    if not self._use_lstm:
      return AgentState(tf.zeros([batch_size, 0], dtype=tf.float32),
                        frame_stacking_state=initial_frame_stacking_state(
            self._stack_size, batch_size, self._observation_space['image']))
    return AgentState(
        core_state=self._core.get_initial_state(
            batch_size=batch_size, dtype=tf.float32),
        frame_stacking_state=initial_frame_stacking_state(
            self._stack_size, batch_size, self._observation_space['image']))

  def _torso(self, prev_action, env_output):
    # [batch_size, output_units]
    obs = env_output.observation
    image = obs['image']
    conv_out = self._body(image)
    
    if self._lang_key == 'none':
      lang = tf.zeros((conv_out.shape[0], 0), dtype=tf.float32)
    else:
      token = obs[self._lang_key]
      if self._lang_key == 'token':
        # One-hot encoding
        token = tf.cast(token, tf.int32)
        token = tf.one_hot(token, self._vocab_size)
      lang = self._mlp(token)
    
    
    # [batch_size, num_actions]
    one_hot_prev_action = tf.one_hot(prev_action, self._num_actions)
    # [batch_size, torso_output_size]
    return tf.concat(
        [conv_out, lang, tf.expand_dims(env_output.reward, -1), one_hot_prev_action],
        axis=1)

  def _head(self, core_output):
    # [batch_size, 1]
    value = self._value(core_output)

    # [batch_size, num_actions]
    advantage = self._advantage(core_output)
    advantage -= tf.reduce_mean(advantage, axis=-1, keepdims=True)

    # [batch_size, num_actions]
    q_values = value + advantage

    action = tf.cast(tf.argmax(q_values, axis=1), tf.int32)
    aux_outputs = self._aux_head(core_output, action)
    return self.AgentOutput(action, q_values, *aux_outputs)

  def _aux_head(self, core_output, action):
    if len(self._aux_pred_heads) == 0:
      return []
    input_ = tf.concat([core_output, tf.one_hot(action, self._num_actions)], axis=1)
    trunk_output = self._aux_trunk(input_)
    aux_outputs = []
    for pred_head in self._aux_pred_heads:
      aux_outputs.append(self.__getattribute__('_pred_{}'.format(pred_head))(trunk_output))
    return aux_outputs

  def __call__(self, input_, agent_state, unroll=False):
    """Applies a network mapping observations to actions.

    Args:
      input_: A pair of:
        - previous actions, <int32>[batch_size] tensor if unroll is False,
          otherwise <int32>[time, batch_size].
        - EnvOutput, where each field is a tensor with added front
          dimensions [batch_size] if unroll is False and [time, batch_size]
          otherwise.
      agent_state: AgentState with batched tensors, corresponding to the
        beginning of each unroll.
      unroll: should unrolling be aplied.

    Returns:
      A pair of:
        - outputs: AgentOutput, where action is a tensor <int32>[time,
            batch_size], q_values is a tensor <float32>[time, batch_size,
            num_actions]. The time dimension is not present if unroll=False.
        - agent_state: Output AgentState with batched tensors.
    """
    if not unroll:
      # Add time dimension.
      input_ = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                     input_)
    prev_actions, env_outputs = input_
    outputs, agent_state = self._unroll(prev_actions, env_outputs, agent_state)
    if not unroll:
      # Remove time dimension.
      outputs = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

    import numpy as np
    try:
      num_params = sum([np.prod(v.shape) for v in self.trainable_variables])

      # Print the number of parameters
      print("!"* 1000)
      print(f'Total number of parameters: {num_params}')
      if hasattr(self, '_embedding'):
        print(f'Number of embedding params: {sum([np.prod(v.shape) for v in self._embedding.trainable_variables])}')
      print(f'Number of {"lstm" if self._use_lstm else "mlp core"} params: {sum([np.prod(v.shape) for v in self._core.trainable_variables])}')
      if hasattr(self, '_mlp'):
        print(f'Number of mlp params: {sum([np.prod(v.shape) for v in self._mlp.trainable_variables])}')
      print(f'Number of cnn params: {sum([np.prod(v.shape) for v in self._body.trainable_variables])}')
      print(f'Number of policy_logits params: {sum([np.prod(v.shape) for v in self._advantage.trainable_variables])}')
      print(f'Number of baseline params: {sum([np.prod(v.shape) for v in self._value.trainable_variables])}')
      if hasattr(self, '_embedding'):
        print(f'Number of params not in embedding: {num_params - sum([np.prod(v.shape) for v in self._embedding.trainable_variables])}')
    except Exception as e:
      print(f'Error printing param count: {e}')

    return outputs, agent_state

  def _unroll(self, prev_actions, env_outputs, agent_state):
    # [time, batch_size, <field shape>]
    unused_reward, done, observation, _, _ = env_outputs
    image = observation['image']
    image = tf.cast(image, tf.float32)

    initial_agent_state = self.initial_state(batch_size=tf.shape(done)[1])

    stacked_frames, frame_state = stack_frames(
        image, agent_state.frame_stacking_state, done, self._stack_size)
    
    # Make a tensorflow copy of the observation
    observation = {k: tf.identity(v) for k, v in observation.items()}
    if self._uses_int_input:
      stacked_frames = stacked_frames / 255
    observation['image'] = stacked_frames
    env_outputs = env_outputs._replace(observation=observation)
    core_state = agent_state.core_state
    # [time, batch_size, torso_output_size]
    torso_outputs = utils.batch_apply(self._torso, (prev_actions, env_outputs))

    if not self._use_lstm:
      core_outputs = utils.batch_apply(self._core, (torso_outputs,))
    else:
      core_outputs, core_state = _unroll_cell(
          torso_outputs, done, agent_state.core_state,
          initial_agent_state.core_state,
          self._core)

    agent_output = utils.batch_apply(self._head, (core_outputs,))
    return agent_output, AgentState(core_state, frame_state)
