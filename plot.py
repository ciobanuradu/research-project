import ast
import os
import sys
import re
from os import listdir,path
from os.path import isfile, join
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt





# data = pd.read_csv('results.csv')
# print(data["returns"])
# X = [i * 5000 for i in range(len(data["returns"]))]

# plt.plot(X, -1 * data["returns"])
# plt.xlabel('Training Steps')
# plt.ylabel('Average Return')
# plt.title('CDQN Training')
# plt.show()

indices = [0, 2, 4, 5, 6, 7, 8]

data_continuous_first = [pd.read_csv(os.path.join(os.curdir, 'results_experiment', f'results_{2 * i + 1}.csv')) for i in indices]
data_continuous_second = [pd.read_csv(os.path.join(os.curdir, 'final_experiment_run', f'results_{i}.csv')) for i in indices]
data_drift = [pd.read_csv(os.path.join(os.curdir, 'results_experiment', f'results_{2 * i}.csv')) for i in indices]
data_induced_decoherence = [pd.read_csv(os.path.join(os.curdir, 'results_experiment_2', f'results_1{i}.csv')) for i in indices]

returns_continuous_first = [ast.literal_eval(data["returns"][0]) for data in data_continuous_first]
returns_continuous_first = [[float(x) for x in return_continuous_first] for return_continuous_first in returns_continuous_first]
errors_continuous_first = [ast.literal_eval(data["errors"][0]) for data in data_continuous_first]
errors_continuous_first = [[float(x) for x in error_continuous_first] for error_continuous_first in errors_continuous_first]

returns_continuous_second = [ast.literal_eval(data["returns"][0]) for data in data_continuous_second]
returns_continuous_second = [[float(x) for x in return_continuous_second] for return_continuous_second in returns_continuous_second]
errors_continuous_second = [ast.literal_eval(data["errors"][0]) for data in data_continuous_second]
errors_continuous_second = [[float(x) for x in error_continuous_second] for error_continuous_second in errors_continuous_second]


returns_induced_decoherence = [ast.literal_eval(data["returns"][0]) for data in data_induced_decoherence]
returns_induced_decoherence = [[float(x) for x in return_induced_decoherence] for return_induced_decoherence in returns_induced_decoherence]
errors_induced_decoherence = [ast.literal_eval(data["errors"][0]) for data in data_induced_decoherence]
errors_induced_decoherence = [[float(x) for x in error_induced_decoherence] for error_induced_decoherence in errors_induced_decoherence]

returns_drift = [ast.literal_eval(data["returns"][0]) for data in data_drift]
returns_drift = [[float(x) for x in return_drift] for return_drift in returns_drift]
errors_drift = [ast.literal_eval(data["errors"][0]) for data in data_drift]
errors_drift = [[float(x) for x in error_drift] for error_drift in errors_drift]

for i in range(7):
    plt.ylim(0, 220)
    plt.errorbar([500 * k for k in range(len(returns_continuous_first[i]))], [-1 * j for j in returns_continuous_first[i]], yerr=([err / 10 for err in errors_continuous_first[i]]), label=f'Continuous low gamma', color="blue")
    # plt.errorbar([500 * k for k in range(len(returns_continuous_second[i]))], [-1 * j for j in returns_continuous_second[i]], yerr=([err / 10 for err in errors_continuous_second[i]]), label=f'Continuous high gamma', color="orange")
    plt.errorbar([500 * k for k in range(len(returns_drift[i]))], [-1 * j for j in returns_drift[i]], yerr=([err / 10 for err in errors_drift[i]]), label=f'Parameter drift', color="green")
    plt.errorbar([500 * k for k in range(len(returns_induced_decoherence[i]))], [-1 * j for j in returns_induced_decoherence[i]], yerr=([err / 10 for err in errors_induced_decoherence[i]]), label=f'Induced decoherence', color="red")
    plt.legend()
    plt.ylabel("Average time")
    plt.xlabel("Number of steps trained")
    plt.title(f"Evaluation during training of model nr. {i}")
    plt.show()



# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# print(returns_continuous[0][0])

    
# plt.legend([f'Continuous {i}' for i in range(8)])
# plt.show()