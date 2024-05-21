import matplotlib.pyplot as plt
import numpy as np

# model names
models = ['Llama 3 8B', 'Mistral 7B', 'Phi-3 3.8B']

# data
llama_data = [10, 57, 62]  
mistral_data = [199, 121, 135]  
phi_data = [543, 521, 512]  

# x-axis position
x = np.arange(len(models))

# bar width
width = 0.3

# plot 
fig, ax = plt.subplots()
ax.set_facecolor('whitesmoke')
bars1 = ax.bar(x - width, llama_data, width, label='min', alpha=0.85, edgecolor='grey', color='xkcd:jade', zorder=2.6)
bars2 = ax.bar(x, mistral_data, width, label='median', alpha=0.85, edgecolor='grey', color='xkcd:bordeaux', zorder=2.6)
bars3 = ax.bar(x + width, phi_data, width, label='max', alpha=0.85, edgecolor='grey', color='xkcd:azure', zorder=2.6)

plt.grid(axis='y',zorder=1)
plt.ylim(0, 600)

# labels and title
ax.set_ylabel('tokens per answer')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# bar labels
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)

plt.show()
