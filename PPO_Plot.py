import tensorflow as tf
import pandas as pd
import tensorboard.backend.event_processing.event_accumulator as event_accumulator
import matplotlib.pyplot as plt


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

plt.plot(X, data)
plt.xlabel('Training Steps')
plt.ylabel('Average Return')
plt.title('CDQN Training')
plt.show()