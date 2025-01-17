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


die () {
    echo >&2 "$@"
    exit 1
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

ENVIRONMENT=$1
AGENT=$2
NUM_ACTORS=$3
ENV_BATCH_SIZE=$4
WANDB_API_KEY=$5
GPU=$6
EXPID=$7
shift 7
args="$@"
echo "All arguments: $args"

if [[ "$ENVIRONMENT" == "messenger" ]]; then
    export ENVIRONMENT="messenger_env"
fi

export PYTHONPATH=$PYTHONPATH:/

ACTOR_BINARY="WANDB_API_KEY=${WANDB_API_KEY} CUDA_VISIBLE_DEVICES='' python3 ../${ENVIRONMENT}/${AGENT}_main.py --run_mode=actor --exp_name=L${EXPID}";
LEARNER_BINARY="CUDA_VISIBLE_DEVICES=${GPU} python3 ../${ENVIRONMENT}/${AGENT}_main.py --run_mode=learner --exp_name=L${EXPID}";
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NUM_ENVS=$(($NUM_ACTORS*$ENV_BATCH_SIZE))


tmux new-session -d -t seed_rl
mkdir -p /tmp/seed_rl
cat >/tmp/seed_rl/instructions <<EOF
Welcome to the SEED local training of ${ENVIRONMENT} with ${AGENT}.
SEED uses tmux for easy navigation between different tasks involved
in the training process. To switch to a specific task, press CTRL+b, [tab id].
You can stop training at any time by executing 'stop_seed'
EOF
tmux send-keys "alias stop_seed='/seed_rl/stop_local.sh seed_rl'" ENTER
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
tmux send-keys -t "learner" "$COMMAND" ENTER

echo "NUM_ACTORS: ${NUM_ACTORS}"
for ((id=0; id<$NUM_ACTORS; id++)); do
    tmux new-window -d -n "actor_${id}"
    COMMAND=''"${ACTOR_BINARY}"' --logtostderr --pdb_post_mortem '"$args"' --num_envs='"${NUM_ENVS}"' --task='"${id}"' --env_batch_size='"${ENV_BATCH_SIZE}"''
    tmux send-keys -t "actor_${id}" "$COMMAND" ENTER
done

tmux new-window -d -n tensorboard
mkdir /logs
tmux send-keys -t "tensorboard" "tensorboard --logdir /logs/ --port 6$EXPID --bind_all" ENTER

tmux attach -t seed_rl
