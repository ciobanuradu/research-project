import os
import sys
import re
import tensorflow as tf
from os import listdir,path
from os.path import isfile, join
import numpy as np
import pandas as pd
import ast
from matplotlib import pyplot as plt

# read all output files
def read_output_files():
    results = []

    for i in range(288):
        filename = f'results_{i}.csv'
        filepath = os.path.join(os.curdir, 'results', filename)
        data = pd.read_csv(filepath)
        returns = ast.literal_eval(data["losses"][0])
        returns = [-float(x) for x in returns]
        min_return = min(returns)
        mean_return = tf.reduce_mean(returns)
        results.append((data, min_return, mean_return))

    # get 15 minimum returns

    # print(sorted_min_results)

    # get 15 minimum average returns

    sorted_min_results_indices = []
    sorted_mean_results_indices = []
    for i, result in enumerate(results):
        if len(sorted_min_results_indices) < 15:
            sorted_min_results_indices.append(i)
            sorted_min_results_indices.sort(key=lambda x: results[x][1])
        else:
            sorted_min_results_indices.sort(key=lambda x: results[x][1])
            if result[1] < results[sorted_min_results_indices[-1]][1]:
                sorted_min_results_indices[-1] = i
        if len(sorted_mean_results_indices) < 15:
            sorted_mean_results_indices.append(i)
            sorted_mean_results_indices.sort(key=lambda x: results[x][2])
        else:
            sorted_mean_results_indices.sort(key=lambda x: results[x][2])
            if result[2] < results[sorted_mean_results_indices[-1]][2]:
                sorted_mean_results_indices[-1] = i
                
    combined_indices = sorted_min_results_indices + sorted_mean_results_indices
    combined_indices = [x for x in set(combined_indices)]
    configs = [results[i][0] for i in combined_indices]


    #combine the two, and remove duplicates
    configs = [{'EPSILON_MAX': float(x['EPSILON_MAX']), 
                'EPSILON_MIN': float(x['EPSILON_MIN']), 
                'EPSILON_DECAY_STEPS': float(x['EPSILON_DECAY_STEPS']), 
                'learning_rate': float(x['learning_rate']), 
                'gamma': float(x['gamma']), 
                'n_step_update': float(x['n_step_update'])} for x in configs]
    return configs

def read_experiment_files():
    results = []

    for i in range(288):
        filename = f'results_{i}.csv'
        filepath = os.path.join(os.curdir, 'results', filename)
        data = pd.read_csv(filepath)
        returns = ast.literal_eval(data["losses"][0])
        returns = [-float(x) for x in returns]
        min_return = min(returns)
        mean_return = tf.reduce_mean(returns)
        results.append((data, min_return, mean_return))

    # get 15 minimum returns

    # print(sorted_min_results)

    # get 15 minimum average returns

    sorted_min_results_indices = []
    sorted_mean_results_indices = []
    for i, result in enumerate(results):
        if len(sorted_min_results_indices) < 15:
            sorted_min_results_indices.append(i)
            sorted_min_results_indices.sort(key=lambda x: results[x][1])
        else:
            sorted_min_results_indices.sort(key=lambda x: results[x][1])
            if result[1] < results[sorted_min_results_indices[-1]][1]:
                sorted_min_results_indices[-1] = i
        if len(sorted_mean_results_indices) < 15:
            sorted_mean_results_indices.append(i)
            sorted_mean_results_indices.sort(key=lambda x: results[x][2])
        else:
            sorted_mean_results_indices.sort(key=lambda x: results[x][2])
            if result[2] < results[sorted_mean_results_indices[-1]][2]:
                sorted_mean_results_indices[-1] = i
                
    combined_indices = sorted_min_results_indices + sorted_mean_results_indices
    combined_indices = [x for x in set(combined_indices)]
    configs = [results[i][0] for i in combined_indices]


    #combine the two, and remove duplicates
    configs = [{'EPSILON_MAX': float(x['EPSILON_MAX']), 
                'EPSILON_MIN': float(x['EPSILON_MIN']), 
                'EPSILON_DECAY_STEPS': float(x['EPSILON_DECAY_STEPS']), 
                'learning_rate': float(x['learning_rate']), 
                'gamma': float(x['gamma']), 
                'n_step_update': float(x['n_step_update'])} for x in configs]
    return configs

# minreturn = -1
# minindex = -1
# for i in range(288):
#     filename = f'results_{i}.csv'
#     filepath = os.path.join(os.curdir, 'results', filename)
#     data = pd.read_csv(filepath)
#     returns = ast.literal_eval(data["losses"][0])
#     returns = [-float(x) for x in returns]
#     min_return = min(returns)
#     if minindex == -1:
#         minindex = i
#         minreturn = min_return
#     if min_return < minreturn:
#         minreturn = min_return
#         minindex = i

# print(minindex)
# print(minreturn)

outs = read_output_files()
for i in range(10):
    print(outs[i])