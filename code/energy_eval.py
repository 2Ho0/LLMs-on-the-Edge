import pandas as pd
import matplotlib.pyplot as plt

#file_name = 'C:/Users/Admin/Desktop/LLMs on the Edge/energy_eval_data/llama3.xlsx'
#file_name = 'C:/Users/Admin/Desktop/LLMs on the Edge/energy_eval_data/mistral.xlsx'
file_name = 'C:/Users/Admin/Desktop/LLMs on the Edge/energy_eval_data/phi3.xlsx'
df = pd.read_excel(file_name)

# get energy values
values = df.iloc[:, 3].values

# plot
plt.figure(figsize=(16, 6))
ax = plt.subplot(111)
ax.set_facecolor('whitesmoke')
plt.axhline(y=3.4, color='r', linestyle='-', label="baseline", alpha=0.5)
plt.plot(range(0,2962), values, label='Phi-3 3.8B', color='xkcd:jade', alpha=0.85)
#plt.plot(range(0,6409), values, label='Llama 3 8B', color='xkcd:bordeaux', alpha=0.85)
#plt.plot(range(0,4835), values, label='Mistral 7B', color='xkcd:azure', alpha=0.85)

# Setting labels and limits
plt.xlabel('time (s)')
plt.ylabel('energy (W)')
plt.xlim(0, 6500)
plt.ylim(0, 12)

# Adding grid
plt.grid(True)

# Show plot
plt.show()
