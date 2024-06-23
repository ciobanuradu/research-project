import tensorboard.backend.event_processing.event_accumulator as event_accumulator
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
import ast

filename = f'results_227.csv'
filepath = os.path.join(os.curdir, 'results', filename)
data = pd.read_csv(filepath)
returns = ast.literal_eval(data["losses"][0])
dqn_returns = [-float(x) for x in returns]

# read from summary writer
event_acc = event_accumulator.EventAccumulator("ppo/eval/events.out.tfevents.1716902380.DESKTOP-B5PDINK.2086.1.v2",
                             size_guidance={event_accumulator.TENSORS: 0}
                            )
event_acc.Reload()

df = (pd.DataFrame(event_acc.Tensors('Metrics/AverageEpisodeLength')))
ret = []
for i, (w, s, t) in df.iterrows():
    ret += [[w, s, tf.make_ndarray(t)]]

print(ret)

data = [float(x[2]) for x in ret]
X = [x[1] for x in ret]

print(X)
print(data)

plt.plot(X, data, label='PPO')
plt.plot([500 * i for i in range(len(dqn_returns))], dqn_returns, label='DQN')
plt.xlabel('Training Steps')
plt.ylabel('Average Evaluation Episode Length')
plt.legend()
plt.title('PPO vs DQN Performance During Training')
plt.show()

