import numpy as np
import pandas as pd
import os

from sklearn.model_selection import ParameterGrid
from cdqn_agent_function import train

import random

    # EPSILON_MAX = parameters["EPSILON_MAX"] # Max exploration rate
    # EPSILON_MIN = parameters["EPSILON_MIN"] # Min exploration rate
    # EPSILON_DECAY_STEPS = parameters["EPSILON_DECAY_STEPS"] # How many steps to decay from max exploration to min exploration
    
    # learning_rate = parameters["learning_rate"]  # @param {type:"number"}
    # gamma = parameters["gamma"]
    # n_step_update = parameters["n_step_update"]  # @param {type:"integer"}

params = {
    "EPSILON_MAX":[1.0, 0.8, 0.5],
    "EPSILON_MIN":[0.01, 0.05, 0.1, 0.2],
    "EPSILON_DECAY_STEPS":[10000, 25000, 50000, 100000],
    "learning_rate":[1e-4, 1e-5, 1e-6],
    "gamma":[0.99, 0.9, 0.8, 0.6],
    "n_step_update":[2]
}


paramgrid = list(ParameterGrid(params))
random.shuffle(paramgrid)

for i, config in enumerate(paramgrid):
    print(f'Iteration {i+1}/{len(paramgrid)}')
    print(config)
    returns, losses = train(config)
    config['returns'] = returns
    config['losses'] = losses
    df = pd.DataFrame([config])
    filename = f'results_{i}.csv'
    filepath = os.path.join(os.curdir, 'results', filename)
    df.to_csv(filepath)