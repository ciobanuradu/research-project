from __future__ import absolute_import, division, print_function

from continuous_environment import continuous_environment
from dynamic_environment import dynamic_environment
from discrete_environment import discrete_environment
from alt_dynamic_environment import alt_dynamic_environment

import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import base64
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import reverb

import abc
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


from tf_agents.networks import categorical_q_network

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import TimeLimit
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
import math
import time
import pandas

def optimal_solution_for_2_actions(p1, f1, p2, f2, gamma, fthresh):
  pmax = max(p1, p2)
  N1 = np.floor(- 1 / gamma * np.log((fthresh - 1/4) / (f1 - 1/4)))
  N2 = np.floor(- 1 / gamma * np.log((fthresh - 1/4) / (f2 - 1/4)))
  return min((1/p1 + (1 - (1 - pmax) ** (N1 - 1)/pmax + ((1 - pmax) ** (N1 - 1)) * (N1 - 1))) / (1 - ((1 -pmax) ** (N1 - 1))), 
             (1/p2 + (1 - (1 - pmax) ** (N2 - 1)/pmax + ((1 - pmax) ** (N2 - 1)) * (N2 - 1))) / (1 - ((1 -pmax) ** (N2 - 1))))

#HYPERPARAMS

num_iterations = 300000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 512  # @param {type:"integer"}
learning_rate = 1e-5  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 100  # @param {type:"integer"}
eval_interval = 500  # @param {type:"integer"}
checkpoint_interval = 50000 # @param {type:"integer"}

num_atoms = 51  # @param {type:"integer"}

min_q_value = -1200  # @param {type:"integer"}
max_q_value = 0  # @param {type:"integer"}

n_step_update = 2  # @param {type:"integer"}

MAX_LOSS = 120



EPSILON_MAX = 1.0 # Max exploration rate
EPSILON_MIN = 0.01 # Min exploration rate
EPSILON_DECAY_STEPS = 40000 # How many steps to decay from max exploration to min exploration



# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
    num_units,
    activation=tf.keras.activations.relu,
    kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_in', distribution='truncated_normal'))

checkpoint_dir = os.path.join(os.path.curdir, 'models','cdqn')
policy_dir = os.path.join(os.path.curdir, 'actual_experiment_3', 'policies')
results_path = os.path.join(os.path.curdir, 'results.csv')

def train(parameters, index = None):
    
    policy_dir = os.path.join(os.path.curdir, 'actual_experiment_3', 'policies')
    if index is not None:
        policy_dir = os.path.join(policy_dir, str(index))
    
    
    EPSILON_MAX = parameters["EPSILON_MAX"] # Max exploration rate
    EPSILON_MIN = parameters["EPSILON_MIN"] # Min exploration rate
    EPSILON_DECAY_STEPS = parameters["EPSILON_DECAY_STEPS"] # How many steps to decay from max exploration to min exploration 
    
    learning_rate = parameters["learning_rate"]  # @param {type:"number"}
    EPSILON_DECAY_STEPS *= 1e-4 // learning_rate
    EPSILON_DECAY_STEPS = int(np.floor(EPSILON_DECAY_STEPS))
    gamma = parameters["gamma"]
    n_step_update = parameters["n_step_update"]  # @param {type:"integer"}

    num_iterations = 5 * (EPSILON_DECAY_STEPS)
    num_iterations = int(num_iterations)

    actions = [[0.4, 0.6],[0.2, 1]] #[[p1,F1],[p2,F2]]

    #TODO: TEMPORARY FOR SANITY CHECK
    # actions =[[0.5000009999999999, 0.499999], [0.5208227669187396, 0.47917723308126037], [0.5433787177479525, 0.45662128225204746], [0.5678132875803512, 0.43218671241964884], [0.5942829410839893, 0.40571705891601073], [0.6229571744103175, 0.37704282558968244], [0.6540196005482233, 0.34598039945177667], [0.6876691250740252, 0.3123308749259747], [0.7241212198262379, 0.27587878017376216], [0.7636093026609719, 0.2363906973390281]]
    # actions = [action[::-1] for action in actions]

    # actions = [[0.4, 0.6],[0.2, 1]]
    links = 4
    threshold = 0.5
    decay_rate = 0.2 # NOTE: Fiddle with this dial in the future, default: 0.2
    alpha = 15
    timeout = 1200
    mmin = 0.1
    mmax = 0.25
    ttime = 125

    print("timeout:", timeout)

    def gamma_function(x):
        if x < 0.33:
            return x + (0.33 - 0.1) / 80
        else:
            return x
        
    def alt_gamma_function(x):
        timestep = x / links * (0.23) + 0.1

    ENVIRONMENT_TYPE = parameters["ENVIRONMENT_TYPE"]

    # discrete_eval_env = tf_py_environment.TFPyEnvironment(TimeLimit(discrete_environment(actions, decay_rate, threshold, links), duration=timeout))

    env = None
    if ENVIRONMENT_TYPE == "continuous":
        env = continuous_environment(actions, decay_rate, threshold, links)
    elif ENVIRONMENT_TYPE == "discrete":
        env = discrete_environment(actions, decay_rate, threshold, links)
    elif ENVIRONMENT_TYPE == "dynamic":
        env = dynamic_environment(actions, decay_rate, threshold, links, min=mmin, max=mmax, time=ttime)
    elif ENVIRONMENT_TYPE == "dynamic2":
        env = alt_dynamic_environment(actions, decay_rate, threshold, links, min=mmin, max=mmax)
        
    env.reset()

    print(env.actions)

    # print("optimal expected time:", optimal_solution_for_2_actions(0.3, 0.7, 0.6, 0.5, 0.1, 0.3))

    train_py_env = None
    eval_py_env = None
    
    if ENVIRONMENT_TYPE == "continuous":
        train_py_env = continuous_environment(actions, decay_rate, threshold, links)
        train_py_env = TimeLimit(train_py_env, duration=timeout)
        eval_py_env = continuous_environment(actions, decay_rate, threshold, links)
        eval_py_env = TimeLimit(eval_py_env, duration=timeout)
    elif ENVIRONMENT_TYPE == "discrete":
        train_py_env = discrete_environment(actions, decay_rate, threshold, links)
        train_py_env = TimeLimit(train_py_env, duration=timeout)
        eval_py_env = discrete_environment(actions, decay_rate, threshold, links)
        eval_py_env = TimeLimit(eval_py_env, duration=timeout)
    elif ENVIRONMENT_TYPE == "dynamic":
        train_py_env = dynamic_environment(actions, decay_rate, threshold, links, min=mmin, max=mmax, time=ttime)
        train_py_env = TimeLimit(train_py_env, duration=timeout)
        # train_py_env = parallel_py_environment.ParallelPyEnvironment([
        #     lambda: TimeLimit(dynamic_environment(actions, decay_rate, threshold, links, min=0.1, max=0.33, time=80), duration=timeout)] * 4)
        eval_py_env = dynamic_environment(actions, decay_rate, threshold, links, min=mmin, max=mmax, time=ttime)
        eval_py_env = TimeLimit(eval_py_env, duration=timeout)
        # eval_py_env = parallel_py_environment.ParallelPyEnvironment([
        #     lambda: TimeLimit(dynamic_environment(actions, decay_rate, threshold, links, min=0.1, max=0.33, time=80), duration=timeout)] * 4)
    elif ENVIRONMENT_TYPE == "dynamic2":
        train_py_env = alt_dynamic_environment(actions, decay_rate, threshold, links, min=mmin, max=mmax)
        train_py_env = TimeLimit(train_py_env, duration=timeout)
        # train_py_env = parallel_py_environment.ParallelPyEnvironment([
        #     lambda: TimeLimit(alt_dynamic_environment(actions, decay_rate, threshold, links, min=0.1, max=0.33), duration=timeout)] * 4)
        eval_py_env = alt_dynamic_environment(actions, decay_rate, threshold, links, min=mmin, max=mmax)
        eval_py_env = TimeLimit(eval_py_env, duration=timeout)
        # eval_py_env = parallel_py_environment.ParallelPyEnvironment([
        #     lambda: TimeLimit(alt_dynamic_environment(actions, decay_rate, threshold, links, min=0.1, max=0.33), duration=timeout)] * 4)
        
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    print('Observation Spec:')
    print(train_env.time_step_spec().observation)

    fc_layer_params = (128, 16, 16)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1


    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    # dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    # q_values_layer = tf.keras.layers.Dense(
    #     num_actions,
    #     activation=None,
    #     kernel_initializer=tf.keras.initializers.RandomUniform(
    #         minval=-0.03, maxval=0.03),
    #     bias_initializer=tf.keras.initializers.Constant(-0.2))
    # q_net = sequential.Sequential(dense_layers + [q_values_layer])

    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # train_step_counter = tf.Variable(0)

    # agent = dqn_agent.DqnAgent(
    #     train_env.time_step_spec(),
    #     train_env.action_spec(),
    #     q_network=q_net,
    #     optimizer=optimizer,
    #     td_errors_loss_fn= common.element_wise_squared_loss,
        # train_step_counter=train_step_counter)

    # agent.initialize()

    categorical_q_net = categorical_q_network.CategoricalQNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        num_atoms=num_atoms,
        fc_layer_params=fc_layer_params,
        activation_fn=tf.keras.activations.relu)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    epsilon = tf.compat.v1.train.polynomial_decay(
        EPSILON_MAX,
        global_step,
        EPSILON_DECAY_STEPS,
        end_learning_rate=EPSILON_MIN)

    agent = categorical_dqn_agent.CategoricalDqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        epsilon_greedy=epsilon,
        categorical_q_network=categorical_q_net,
        optimizer=optimizer,
        min_q_value=min_q_value,
        max_q_value=max_q_value,
        n_step_update=n_step_update,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gamma,
        train_step_counter=global_step)
    agent.initialize()



    eval_policy = agent.policy
    collect_policy = agent.collect_policy



    def compute_avg_return(environment, policy, num_episodes=10):
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
            returns.append(episode_return)
            if i % 10 == 9:
                print(f"episodes {i - 9} - {i} elapsed time: {time.time() - start}")
                start = time.time()
                if total_return / i < -1 * (timeout - 1):
                    steps = i + 1
                    break
        if steps == 0:
            steps = num_episodes
        avg_return = total_return / steps
        # print("returns:", returns)
        sigma = np.std(returns)
        return avg_return.numpy()[0], sigma

    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    print("starting replay server")

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=2,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    replay_buffer.py_client,
    table_name,
    sequence_length=2)

    print("replay buffer set up")

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(
        random_policy, use_tf_function=True),
        [rb_observer],
        max_steps=initial_collect_steps).run(train_py_env.reset())

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    print("driver instantiated")

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)


    print("evaluating baseline...")
    # print("evaluating policy once")
    # Evaluate the agent's policy once before training.
    avg_return, sigma = compute_avg_return(eval_env, agent.policy, 10)
    
    returns = [[avg_return, sigma]]

    # Reset the environment.
    time_step = train_py_env.reset()

    print("starting driver...")
    # Create a driver to collect experience.
    # collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    #         train_env,
    #         py_tf_eager_policy.PyTFEagerPolicy(
    #         agent.collect_policy, use_tf_function=True),
    #         [rb_observer],
    #         num_episodes=4,
    #     )
    collect_driver = py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=collect_steps_per_iteration)

    print("starting checkpoint")

    # train_checkpointer = common.Checkpointer(
    #     ckpt_dir=checkpoint_dir,
    #     max_to_keep=1,
    #     agent=agent,
    #     policy=agent.policy,
    #     replay_buffer=replay_buffer,
    #     global_step=train_step_counter
    # )

    tf_policy_saver = policy_saver.PolicySaver(agent.policy)

    print("training...")
    losses = []
    min = np.inf
    steps = 0
    for _ in range(num_iterations):
        # Collect a few steps and save to the replay buffer.
        time_step, _ = collect_driver.run(time_step)
        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss
        steps += 1
        if train_loss < min:
            min = train_loss
            steps = 0
        if steps > 5000:
            print("early stopping")
            break

        step = agent.train_step_counter.numpy()
        

        if step % log_interval == 0:
            losses.append(train_loss)
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            print("evaluating policy...")
            
            avg_return, sigma = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            stderror = sigma / math.sqrt(num_eval_episodes)
            # dreturn, dsigma = compute_avg_return(discrete_eval_env, agent.policy, num_eval_episodes)
            
            print('step = {0}: Average Return = {1}: sterror = {2}'.format(step, avg_return, stderror))
            # print("discrete return:", dreturn)
            
            returns.append([avg_return, sigma])
        
    # if step % checkpoint_interval == 0:
    #     print("saving checkpoint...")
    #     train_checkpointer.save(global_step)
    #     print("checkpoint saved!")
    
    print("evaluating policy...")
    avg_return, sigma = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    stderror = sigma / math.sqrt(num_eval_episodes)
    
    print('Average Return = {0}: sterror = {1}'.format(avg_return, stderror))
    returns.append([avg_return, sigma])
    
    tf_policy_saver.save(policy_dir)
    return losses, returns

    # df = pandas.DataFrame(data={"returns": returns})
    # df.to_csv(results_path, sep=',',index=False)