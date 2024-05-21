import os
import sys
import re
from os import listdir,path
from os.path import isfile, join
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('CDQN_Returns_2.csv')
print(data["returns"])
X = [i * 5000 for i in range(len(data["returns"]))]
print(X)

plt.plot(X, -1 * data["returns"])
plt.xlabel('Training Steps')
plt.ylabel('Average Return')
plt.title('CDQN Training')
plt.show()