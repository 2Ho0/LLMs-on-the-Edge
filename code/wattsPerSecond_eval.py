import pandas as pd
import matplotlib.pyplot as plt
import os

# data
files = ['results/llama3_energy_50q_PI5.xlsx', 'results/mistral_energy_50q_PI5.xlsx', 'results/phi3_energy_50q_PI5.xlsx']
labels = ['Llama 3 8B', 'Mistral 7B', 'Phi-3 3.8B']

# create plot
plt.figure(figsize=(10, 6))
plt.grid(zorder=1)
ax = plt.subplot(111)
ax.set_facecolor('whitesmoke')
colors = ['xkcd:azure', 'xkcd:bordeaux', 'xkcd:jade']

# plot data from files
for idx, file in enumerate(files):
    # load files
    data = pd.read_excel(file, header=None, names=['seconds', 'volts', 'ampere', 'watts'])
    columnNames = data.iloc[0] 
    data = data[1:] 
    data.columns = columnNames
    label = labels[idx]
    
    # plot data
    plt.plot(data['seconds'], data['watts'], alpha=0.7, color=colors[idx], label=label)

# add title and labels
plt.xlabel('time in seconds')
plt.ylabel('watts')
plt.legend()

# show plot
plt.show()
