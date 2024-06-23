from __future__ import absolute_import, division, print_function

import discrete_environment

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

from dynamic_environment import dynamic_environment
from continuous_environment import continuous_environment
from tf_agents.networks import categorical_q_network

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
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
import pandas

def optimal_solution_for_2_actions(p1, f1, p2, f2, gamma, fthresh):
  pmax = max(p1, p2)
  N1 = np.floor(- 1 / gamma * np.log((fthresh - 1/4) / (f1 - 1/4)))
  N2 = np.floor(- 1 / gamma * np.log((fthresh - 1/4) / (f2 - 1/4)))
  return min((1/p1 + (1 - (1 - pmax) ** (N1 - 1)/pmax + ((1 - pmax) ** (N1 - 1)) * (N1 - 1))) / (1 - ((1 -pmax) ** (N1 - 1))), 
             (1/p2 + (1 - (1 - pmax) ** (N2 - 1)/pmax + ((1 - pmax) ** (N2 - 1)) * (N2 - 1))) / (1 - ((1 -pmax) ** (N2 - 1))))

#HYPERPARAMS

num_iterations = 100000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 1024  # @param {type:"integer"}
learning_rate = 1e-5  # @param {type:"number"}
log_interval = 1000  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 5000  # @param {type:"integer"}
checkpoint_interval = 10000 # @param {type:"integer"}
gamma = 1.01

num_atoms = 51  # @param {type:"integer"}

min_q_value = -1200  # @param {type:"integer"}
max_q_value = 0  # @param {type:"integer"}

n_step_update = 2  # @param {type:"integer"}

MAX_LOSS = 120

actions = [[0.4, 0.6],[0.2, 1]] #[[p1,F1],[p2,F2]]
links = 4
threshold = 0.5
decay_rate = 0.2  # NOTE: Fiddle with this dial in the future, default: 0.2


EPSILON_MAX = 0.8 # Max exploration rate
EPSILON_MIN = 0.01 # Min exploration rate
EPSILON_DECAY_STEPS = 10000 # How many steps to decay from max exploration to min exploration



checkpoint_dir = os.path.join(os.path.curdir, 'models','cdqn')
policy_dir = os.path.join(os.path.curdir, 'policies')
results_path = os.path.join(os.path.curdir, 'results.csv')

alpha = 15
timeout = (1 / (max(actions, key=lambda x: x[1])[0]) * alpha) * (links ** 2)

print("timeout:", timeout)

env = continuous_environment.continuous_environment(actions, decay_rate, threshold, links)
env.reset()

print(env.actions)

# print("optimal expected time:", optimal_solution_for_2_actions(0.3, 0.7, 0.6, 0.5, 0.1, 0.3))

print('Observation Spec:')
print(env.time_step_spec().observation)

train_py_env = continuous_environment.continuous_environment(actions, decay_rate, threshold, links)
train_py_env = TimeLimit(train_py_env, duration=timeout)
eval_py_env = continuous_environment.continuous_environment(actions, decay_rate, threshold, links)
eval_py_env = TimeLimit(eval_py_env, duration=timeout)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

fc_layer_params = (128, 16, 16)
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

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

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

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


# print("evaluating baseline...")
# # print("evaluating policy once")
# # Evaluate the agent's policy once before training.
# avg_return = compute_avg_return(eval_env, agent.policy, 1)
# returns = [avg_return]

returns = []

# Reset the environment.
time_step = train_py_env.reset()

print("starting driver...")
# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
      agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)

print("starting checkpoint")

train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step_counter
)

tf_policy_saver = policy_saver.PolicySaver(agent.policy)

print("training...")

for _ in range(num_iterations):
  # Collect a few steps and save to the replay buffer.
  time_step, _ = collect_driver.run(time_step)
  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss


  step = agent.train_step_counter.numpy()


  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    print("evaluating policy...")
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)
    
  if step % checkpoint_interval == 0:
    print("saving checkpoint...")
    train_checkpointer.save(global_step)
    print("checkpoint saved!")
    
  if step > num_iterations:
    print(step)
    break

tf_policy_saver.save(policy_dir)

df = pandas.DataFrame(data={"returns": returns})
df.to_csv(results_path, sep=',',index=False)