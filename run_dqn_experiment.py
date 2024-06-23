import numpy as np
import pandas as pd
import os

from sklearn.model_selection import ParameterGrid
from cdqn_agent_function import train
from process_grid_search_results import read_output_files


import functools
import os
import time

from absl import app
from absl import flags
from absl import logging

from tf_agents.system import system_multiprocessing as multiprocessing

import random

    # EPSILON_MAX = parameters["EPSILON_MAX"] # Max exploration rate
    # EPSILON_MIN = parameters["EPSILON_MIN"] # Min exploration rate
    # EPSILON_DECAY_STEPS = parameters["EPSILON_DECAY_STEPS"] # How many steps to decay from max exploration to min exploration
    
    # learning_rate = parameters["learning_rate"]  # @param {type:"number"}
    # gamma = parameters["gamma"]
    # n_step_update = parameters["n_step_update"]  # @param {type:"integer"}


# def main(_):
paramgrid = []
configs = read_output_files()

# param = dict(configs[0])
# param["ENVIRONMENT_TYPE"] = "discrete"
# paramgrid.append(param)

configs = configs[0:10]

# for env in ["dynamic", "dynamic2"]:
for env in ["continuous"]:
    for config in configs:
        params = dict(config)
        params["ENVIRONMENT_TYPE"] = env
        paramgrid.append(params)

# configs = [{
#     "EPSILON_MAX":1,
#     "EPSILON_MIN":0.01,
#     "EPSILON_DECAY_STEPS":5000,
#     "learning_rate":1e-5,
#     "gamma":1.1,
#     "n_step_update":2,
#     "ENVIRONMENT_TYPE":"continuous"

# }]

for i, config in enumerate(paramgrid):
    print(f'Iteration {i+1}/{len(paramgrid)}')
    print(config)
    
    
    if config["learning_rate"] < 5e-5:
        print("skipping slow learning config")
        continue
    
    losses, returns = train(config, i)
    sigmas = [retrn[1] for retrn in returns]
    returns = [retrn[0] for retrn in returns]
    config['returns'] = returns
    config['losses'] = losses
    config['errors'] = sigmas
    df = pd.DataFrame([config])
    filename = f'results_{i}.csv'
    filepath = os.path.join(os.curdir, 'final_experiment_run', filename)
    # filepath = os.path.join(os.curdir, 'results_sanity_check', filename)
    df.to_csv(filepath)


# if __name__ == '__main__':
#   multiprocessing.handle_main(functools.partial(app.run, main))