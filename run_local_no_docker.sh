#!/bin/bash
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


set -e
die () {
    echo >&2 "$@"
    exit 1
}
ENVIRONMENTS="atari|dmlab|football|mujoco|messenger|homecook|vln"
AGENTS="r2d2|vtrace|sac|ppo"
[ "$#" -ne 0 ] || die "Usage: run_local.sh [$ENVIRONMENTS] [$AGENTS] [Num. actors]"
echo $1 | grep -E -q $ENVIRONMENTS || die "Supported games: $ENVIRONMENTS"
echo $2 | grep -E -q $AGENTS || die "Supported agents: $AGENTS"
echo $3 | grep -E -q "^((0|([1-9][0-9]*))|(0x[0-9a-fA-F]+))$" || die "Number of actors should be a non-negative integer without leading zeros"
export ENVIRONMENT=$1
export AGENT=$2
export NUM_ACTORS=$3
shift 3
if [[ $1 ]]; then
  echo $1 | grep -E -q "^((0|([1-9][0-9]*))|(0x[0-9a-fA-F]+))$" || die "Number of environments per actor should be a non-negative integer without leading zeros"
  export ENV_BATCH_SIZE=$1
  shift 1
else
  export ENV_BATCH_SIZE=1
fi
export CONFIG=$ENVIRONMENT

export WANDB_API_KEY=$1
export GPU_LIST=$2
# SPlit GPU list into a list of GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPU_LIST"
echo "GPU_ARRAY: ${GPU_ARRAY[@]}"
export EXPID=$3
export ENVS_PER_ACTOR=$4
shift 4
args="$@"
# Last GPU goes to the learner
GPU=${GPU_ARRAY[-1]}
echo "GPU for learner: ${GPU}"
GPU_ARRAY=("${GPU_ARRAY[@]:0:${#GPU_ARRAY[@]}-1}")

source ~/miniconda3/etc/profile.d/conda.sh || echo "conda apparently wasn't where we thought it was"

if [[ "$ENVIRONMENT" == "messenger" ]]; then
    export ENVIRONMENT="messenger_env"
fi

LEARNER_BINARY="WANDB_API_KEY=${WANDB_API_KEY} CUDA_VISIBLE_DEVICES=${GPU} python3 seed_rl/${ENVIRONMENT}/${AGENT}_main.py --run_mode=learner --exp_name=L${EXPID} --logdir ../logs/L${EXPID}";
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NUM_ENVS=$(($NUM_ACTORS*$ENV_BATCH_SIZE))
CONDA_COMMAND="conda activate $CONDA_DEFAULT_ENV || echo 'Failed to activate conda environment'"
SESSION="L${EXPID}"

tmux new-session -d -s $SESSION
mkdir -p /tmp/seed_rl
cat >/tmp/seed_rl/instructions <<EOF
Welcome to the SEED local training of ${ENVIRONMENT} with ${AGENT}.
SEED uses tmux for easy navigation between different tasks involved
in the training process. To switch to a specific task, press CTRL+b, [tab id].
You can stop training at any time by executing 'stop_seed'
EOF
tmux send-keys "$CONDA_COMMAND" ENTER
tmux send-keys "alias stop_seed='seed_rl/stop_local.sh $SESSION'" ENTER
tmux send-keys clear
tmux send-keys KPEnter
tmux send-keys "cat /tmp/seed_rl/instructions"
tmux send-keys KPEnter
tmux send-keys "python3 check_gpu.py 2> /dev/null"
tmux send-keys KPEnter
tmux send-keys "stop_seed"
tmux new-window -d -n learner

COMMAND='rm /tmp/agent -Rf; '"${LEARNER_BINARY}"' --logtostderr --pdb_post_mortem '"$args"' --num_envs='"${NUM_ENVS}"' --env_batch_size='"${ENV_BATCH_SIZE}"''
echo $COMMAND
tmux send-keys -t "learner" "$CONDA_COMMAND && $COMMAND" ENTER

echo "NUM_ACTORS: ${NUM_ACTORS}"
for ((id=0; id<$NUM_ACTORS; id++)); do
    ACTOR_GPU=${GPU_ARRAY[$id % ${#GPU_ARRAY[@]}]}
    ACTOR_BINARY="WANDB_API_KEY=${WANDB_API_KEY} CUDA_VISIBLE_DEVICES=$ACTOR_GPU python3 seed_rl/${ENVIRONMENT}/${AGENT}_main.py --run_mode=actor --exp_name=L${EXPID} --logdir ../logs/L${EXPID}";
    tmux new-window -d -n "actor_${id}"
    COMMAND=''"${ACTOR_BINARY}"' --logtostderr --pdb_post_mortem '"$args"' --num_envs='"${NUM_ENVS}"' --task='"${id}"' --env_batch_size='"${ENV_BATCH_SIZE}"''
    tmux send-keys -t "actor_${id}" "$CONDA_COMMAND && $COMMAND" ENTER
done

tmux new-window -d -n tensorboard
mkdir -p ../logs/L${EXPID}
tmux send-keys -t "tensorboard" "$CONDA_COMMAND && tensorboard --logdir ../logs/L${EXPID} --port 6$EXPID --bind_all" ENTER

tmux attach -t $SESSION
