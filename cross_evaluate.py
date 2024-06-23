from continuous_environment import continuous_environment
from dynamic_environment import dynamic_environment
from alt_dynamic_environment import alt_dynamic_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import TimeLimit
import pandas as pd
import tensorflow as tf
import time
import os
actions = [[0.4, 0.6],[0.2, 1]]
links = 4
threshold = 0.5
decay_rate = 0.1
mmin = 0.1
mmax = 0.25
ttime = 125

dir1 = os.path.join(os.curdir, "actual_experiment", "policies")
dir2 = os.path.join(os.curdir, "actual_experiment_2", "policies")
dir3 = os.path.join(os.curdir, "actual_experiment_3", "policies")

timeout=1000

cont_env_1 = TimeLimit(continuous_environment(actions=actions, gamma=0.1, threshold=threshold, links=links), duration=timeout)
cont_env_1 = tf_py_environment.TFPyEnvironment(cont_env_1)
cont_env_2 = TimeLimit(continuous_environment(actions=actions, gamma=0.25, threshold=threshold, links=links), duration=timeout)
cont_env_2 = tf_py_environment.TFPyEnvironment(cont_env_2)
drift_env = TimeLimit(dynamic_environment(actions=actions, gamma=decay_rate, threshold=threshold, links=links, min=mmin, max=mmax, time=ttime), duration=timeout)
drift_env = tf_py_environment.TFPyEnvironment(drift_env)
induced_decay_env = TimeLimit(alt_dynamic_environment(actions=actions, gamma=decay_rate, threshold=threshold, links=links, min=mmin, max=mmax), duration=timeout)
induced_decay_env = tf_py_environment.TFPyEnvironment(induced_decay_env)

def load_models(index: int): 
    cont_model_1 = tf.saved_model.load(os.path.join(dir1, f'{2 * index + 1}'))
    cont_model_2 = tf.saved_model.load(os.path.join(dir3, f'{index}'))
    drift_model = tf.saved_model.load(os.path.join(dir1, f'{2 * index}'))
    induced_decay_model = tf.saved_model.load(os.path.join(dir2, f'{index}'))
    return cont_model_1, cont_model_2, drift_model, induced_decay_model

indices = [6]

def compute_returns(environment, policy, num_episodes=250):
        start = time.time()
        total_return = 0.0
        returns = []
        steps = 0
        for i in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return
            returns.append(float(episode_return))
            if i % 10 == 9:
                print(f"episodes {i - 9} - {i} elapsed time: {time.time() - start}")
                start = time.time()
                if total_return / i < -1 * (timeout - 1) and i > 30:
                    steps = i + 1
                    returns = [timeout] * num_episodes
                    break
        return returns

env_names = ["Continuous_low_gamma", "Continuous_high_gamma", "parameter_drift", "induced_decoherence"]

for i in indices:
    print(f"evaluating model set {i}")
    c1, c2, PD, id = load_models(i)
    envs = [cont_env_1, cont_env_2, drift_env, induced_decay_env]
    for j, env in enumerate(envs):
        print(f"evaluating environment {j}/4...")
        rc1 = compute_returns(env, c1)
        print(f"25% done...")
        rc2 = compute_returns(env, c2)
        print(f"50% done...")
        rpd = compute_returns(env, PD)
        print(f"75% done...")
        rid = compute_returns(env, id)
        print(f"all models evaluated!")
        
        result = {
            "env": env_names[j],
            "continuous_low_gamma": rc1,
            "continuous_high_gamma": rc2,
            "parameter_drift": rpd,
            "induced_decoherence": rid
        }
        
        df = pd.DataFrame(result)
        
        # ({'name':result.keys(), 'value':result.Values()})
        filename = f"cross_eval_{i}_{j}"
        filepath = os.path.join(os.curdir, "cross_evaluations", filename)
        df.to_csv(filepath)