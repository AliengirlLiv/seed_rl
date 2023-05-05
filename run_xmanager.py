from absl import app

import itertools
from xmanager import xm
from xmanager import xm_local

def main(args):
  with xm_local.create_experiment(experiment_title='seed_rl') as experiment:
    spec = xm.Dockerfile(
      '/home/jessy/olivia/docker_seed',
      '/home/jessy/olivia/docker_seed/seed_rl/docker/Dockerfile.messenger',
    )

    [executable] = experiment.package([
      xm.Packageable(
        executable_spec=spec,
        executor_spec=xm_local.Vertex.Spec(),
      ),
    ])

    requirements = xm.JobRequirements(A100=1)

#    for hyperparameters in trials:
    for seed in [0]:  # TODO: more seeds
      for mlp_key in ["token"]:#, "token_embed"]:
        name = f"testing_messenger_seedrl"
        script = "docker/run.sh"
        experiment.add(xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            env_vars={
              "WANDB_API_KEY": "642b88300b8a5aa08de1fe3271607684835876d8",
              "GOOGLE_CLOUD_BUCKET_NAME": "external-collab-dreamer/dynalang"},
            args=["conda",
                  "run",
                  "--no-capture-output",
                  "-n",
                  "embodied",
                  "sh",
                  script,
                  'messenger',
                  'vtrace',
                  1,
                  32,
                  '642b88300b8a5aa08de1fe3271607684835876d8',
                #   str(seed),  # TODO: add back?
                  f"--logdir=gs://external-collab-dreamer/dynalang/messenger/{name}",
                  ]
      ))

app.run(main)
