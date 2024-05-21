import matplotlib.pyplot as plt

# data
models = ['Mistal 7B', 'Phi-3 3.8B', 'Llama 3 8B']
scores = {'GPT-4o': [0.859, 0.804, 0.746], 'Human': [0.81, 0.797, 0.733]}

# bar colors, width and position
colors = {'GPT-4o': 'xkcd:azure', 'Human': 'xkcd:bordeaux'}
bar_width = 0.35
positions = range(len(models))

# create plot
fig, ax = plt.subplots()

for i, (eval_type, eval_scores) in enumerate(scores.items()):
    bars = ax.bar([p + i * bar_width for p in positions], eval_scores, bar_width,
                   color=colors[eval_type], label=eval_type, alpha=0.85)
    
    # add bar labels
    for j, score in enumerate(eval_scores):
        ax.text(positions[j] + i * bar_width, score, str(score), ha='center', va='bottom')

# labels of axis
ax.set_xticks([p + bar_width / 2 for p in positions])
ax.set_xticklabels(models)
ax.set_ylim(0, 1) 

# settings
ax.set_ylabel('quality of answers')
ax.set_facecolor('whitesmoke')
ax.set_axisbelow(True)
ax.grid()
ax.set_zorder(2)
ax.legend()

plt.show()
