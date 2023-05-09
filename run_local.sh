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
ENVIRONMENTS="atari|dmlab|football|mujoco|messenger|homecook"
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
export GPU=$2
export EXPID=$3
shift 3
args="$@"
echo "OG All arguments: $args"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..
docker build --network=host -t tmp_seed_rl:${CONFIG} -f seed_rl/docker/Dockerfile.${CONFIG} .

docker_version=$(docker version --format '{{.Server.Version}}')
docker run --gpus all -ti -it --network=host -p 6${EXPID}:6${EXPID} \
  -e HOST_PERMS="$(id -u):$(id -g)" \
  -e ENVIRONMENT="$ENVIRONMENT" \
  -e AGENT="$AGENT" \
  -e NUM_ACTORS="$NUM_ACTORS" \
  -e ENV_BATCH_SIZE="$ENV_BATCH_SIZE" \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -e GPU="$GPU" \
  -e EXPID="$EXPID" \
  -e args="$args" \
  --name seed_${EXPID} --rm tmp_seed_rl:${CONFIG} \
  conda run -n embodied --no-capture-output /bin/bash -c 'docker/run.sh $ENVIRONMENT $AGENT $NUM_ACTORS $ENV_BATCH_SIZE $WANDB_API_KEY $GPU $EXPID $args'

container_id=$(docker inspect -f '{{.Id}}' seed_${EXPID})
docker cp container_id:/logs ../logs/L${EXPID}
