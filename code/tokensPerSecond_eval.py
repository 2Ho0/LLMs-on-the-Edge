import pandas as pd
from statistics import median
import matplotlib.pyplot as plt


# data
models = ['Llama 3 8B', 'Mistral 7B', 'Phi-3 3.8B']
number_of_tokens = [1.831, 2.060, 3.514]

# position of bars
bar_positions = range(len(models))

# plot
fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)
ax.set_facecolor('whitesmoke')
bars = plt.bar(bar_positions, number_of_tokens, width=0.5, color=['xkcd:azure', 'xkcd:azure', 'xkcd:azure'], alpha=0.85, edgecolor='grey', zorder=2.6)

plt.grid(axis='y',zorder=1)

# labels
plt.ylabel('tokens per second')
plt.xticks(bar_positions, models)
plt.ylim(0, 4)

# bar labels
for bar, val in zip(bars, number_of_tokens):
    plt.text(bar.get_x() + bar.get_width() / 2, val, str(val), ha='center', va='bottom')

plt.show()
