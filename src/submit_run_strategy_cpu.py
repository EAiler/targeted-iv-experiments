"""Run for multiple hyperparameter values on slurm cluster."""

import json
import os
from copy import deepcopy
from itertools import product
from typing import Text, Sequence, Dict, Any

from absl import app
from absl import flags

flags.DEFINE_string("experiment_name", None,
                    "The name of the experiment (used for output folder).")
flags.DEFINE_string("output_dir", "/home/haicu/elisabath.ailer/Projects/SingleSampleIV/Output/",
                    "Base directory for all results.")
flags.DEFINE_bool("gpu", False, "Whether to use GPUs.")
flags.DEFINE_string("data_scenario", "standard",
                    "The data scenario used.")
flags.DEFINE_string("functional_type", "gradient",
                    "The type of functional used.")
flags.DEFINE_string("xbar_type", "mean", "The type of xbar used.")

# FIXED FLAGS
#flags.DEFINE_float("logcontrast_threshold", 0.7, "Log Contrast Threshold Value.")
#flags.DEFINE_integer("num_runs", 50, "Number of runs for confidence interval computation.")
#flags.DEFINE_list("experiment_", ["Basis", "Linear_Fourier"],
#                  "Define the methods that should be run.")
#flags.DEFINE_bool("use_data_in_folder", False, "Check if data is already available in the folder, if yes,
# then use the pickle files in the folder.")
#flags.DEFINE_string("add_id", "", "additional identifier, if data should be reused for another method etc.")
#flags.DEFINE_string("scenario", "", "Scenario you would like to run.")

# DEFINITION OF PARAMETER GRID
flags.mark_flag_as_required("experiment_name")
FLAGS = flags.FLAGS


# Some values and paths to be set
user = "elisabath.ailer"
project = "SingleSampleIV"
executable = f"/home/haicu/{user}/miniconda3/envs/insufficient_iv/bin/python"  # inserted the respective environment
run_file = f"/home/haicu/{user}/Projects/{project}/Code/src/run_strategy.py"  # MAY NEED TO UPDATE THIS


# Specify the resource requirements *per run*
num_cpus = 4
num_gpus = 1
mem_mb = 16000
max_runtime = "00-12:00:00"


def get_output_name(value_dict: Dict[Text, Any]) -> Text:
  """Get the name of the output directory."""
  name = ""
  for k, v in value_dict.items():
    name += f"_{k}-{v}"
  return name[1:]

  
def get_flag(key: Text, value: Any) -> Text:
   return f' --{key}={value}'


def submit_all_jobs(args: Sequence[Dict[Text, Any]], fixed_flags) -> None:
  """Generate submit scripts and launch them."""
  # Base of the submit file
  base = list()
  base.append(f"#!/bin/bash")
  base.append("")
  base.append(f"#SBATCH -J {project}{'_gpu' if FLAGS.gpu else ''}")
  #base.append(f"#SBATCH -c {num_cpus}")
  base.append(f"#SBATCH --mem={mem_mb}")
  base.append(f"#SBATCH -t {max_runtime}")
  base.append(f"#SBATCH --nice=0")
  if FLAGS.gpu:
    base.append(f"#SBATCH -p cpu_p")
    base.append(f"#SBATCH --qos cpu_normal") #gpu_long
    base.append(f"#SBATCH --gres=gpu:{num_gpus}")
    # base.append(f"#SBATCH --exclude=icb-gpusrv0[1]")  # keep for interactive
    # base.append(f"#SBATCH --exclude=icb-gpusrv0[22-25]")  # keep for interactive
  else:
    base.append(f"#SBATCH -p cpu_p")
    base.append(f"#SBATCH --qos cpu_normal") #gpu_long
    #base.append(f"#SBATCH --qos cpu_reservation")
    #base.append(f"#SBATCH --reservation=rocky_linux_9_test")

  for i, arg in enumerate(args):
    lines = deepcopy(base)
    output_name = get_output_name(arg)

    # Directory for slurm logs
    output_dir = os.path.join(FLAGS.output_dir, FLAGS.experiment_name + "-" + FLAGS.data_scenario + "-" + FLAGS.functional_type + "-" + FLAGS.xbar_type, output_name)
    logs_dir = output_dir   #os.path.join(output_dir, output_name)

    # Create directories if non-existent (may be created by the program itself)
    if not os.path.exists(logs_dir):
      os.makedirs(logs_dir)

    # The output, logs, and errors from running the scripts
    logs_name = os.path.join(logs_dir, "slurm")
    lines.append(f"#SBATCH -o {logs_name}.out")
    lines.append(f"#SBATCH -e {logs_name}.err")

    # Queue job
    lines.append("")
    runcmd = executable
    runcmd += " "
    runcmd += run_file
    # ASSUMING RUNFILE TAKES THESE THREE ARGUMENTS
    runcmd += f' --output_dir {output_dir}'
    runcmd += f' --experiment_name {FLAGS.experiment_name}'
    runcmd += f' --data_scenario {FLAGS.data_scenario}'
    runcmd += f' --functional_type {FLAGS.functional_type}'
    runcmd += f' --xbar_type {FLAGS.xbar_type}'
    
    # Sweep arguments
    for k, v in arg.items():
      runcmd += get_flag(k, v)

    # Fixed arguments
    for k, v in fixed_flags.items():
      runcmd += get_flag(k, v)

    lines.append(runcmd)
    lines.append("")
    print(lines)
    # Now dump the string into the `run_all.sub` file.
    with open("run_job.cmd", "w") as file:
      file.write("\n".join(lines))

    print(f"Submitting {i}...")
    os.system("sbatch run_job.cmd")


def main(_):
  """Initiate multiple runs."""

  sweep = {
    "strategy_type": [
        "adaptive_sampling",
        "explore_then_exploit",
        "continuous_exploration", 
        "random_sampling"
        ],
    "lam_c": [0.01, 0.02, 0.04, 0.05, 0.07],#[0.02, 0.04, 0.05, 0.07, 0.1],#[0.05, 0.07, 0.1],# [0.01, 0.1, 1.0],
    "lam_first": [0.005, 0.01, 0.05, 0.06, 0.07, 0.1], #[0.01, 0.05, 0.1],#[0.005, 0.075, 0.01],#[0.0001, 0.01, 0.1, 0.05], # [1.0, 2.0, 0.01],
    "seed": [4],
    "T": [16],
    "T_exploration": [10],
    "sample_sigma": [1.0],
    "do_sigma_update": ["Y"]
  }

  values = list(sweep.values())
  args = list(product(*values))
  keys = list(sweep.keys())
  args = [{keys[i]: arg[i] for i in range(len(keys))} for arg in args]
  n_jobs = len(args)
  sweep_dir = os.path.join(FLAGS.output_dir, FLAGS.experiment_name + "-" + FLAGS.data_scenario + "-" + FLAGS.functional_type + "-" + FLAGS.xbar_type)

  fixed_flags = {
    "experiment_name": FLAGS.experiment_name,
    }

  # Create directories if non-existentdta
  if not os.path.exists(sweep_dir):
    os.makedirs(sweep_dir)
  
  for arg in args:
    # Create a sub-directory for each sweep entry
    sub_dir = os.path.join(sweep_dir, get_output_name(arg))   #os.path.join(sweep_dir, '_'.join(f'{k}={v}' for k, v in arg.items()))
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    print(f"Store sweep dictionary to {sub_dir}...")
    with open(os.path.join(sub_dir, "sweep.json"), 'w') as fp:
        json.dump(arg, fp, indent=2)

  print(f"Generate all {n_jobs} submit script and launch them...")
  submit_all_jobs(args, fixed_flags)

  print(f"DONE")


if __name__ == "__main__":
  app.run(main)
