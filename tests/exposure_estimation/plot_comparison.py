# %%
import pandas as pd
import matplotlib.pyplot as plt, numpy as np

plt.rc('text', usetex=True)
df = pd.read_csv('tests/results.csv')

# %%
fig = plt.figure()
ax = fig.add_subplot(111)
methods = ('noisy', 'cerman', 'mst_out', 'quick')
labels = ('Corrupted', 'BTF [3]', 'MST $\mathbf{\hat{e}}_\mathrm{WLS}$ (ours)', 'Tiled $\mathbf{\hat{e}}_\mathrm{WLS}$ (ours)')
data = np.empty((len(methods), len(df.gain.unique()), 2))

for i, g in enumerate(df.gain.unique()):
    subset = df[df.gain == g]
    for j, m in enumerate(methods):
    	data[j,i] = subset[m].mean(), subset[m].var()

for i in range(len(data)):
    plt.errorbar(range(data.shape[1]), data[i,:,0], yerr=1.96*np.sqrt(data[i,:,1]/subset.shape[0]), capsize=4, fmt='x-')
    plt.xticks(ticks=range(data.shape[1]), labels=[str(g*100) for g in df.gain.unique()])
    for x, y in zip(np.arange(data.shape[1]), data[i,:,0]):
        plt.text(x-0.05, y, f'{y:.2f}', ha='right', va='top')

plt.minorticks_on()
plt.xlim(left=-0.5)
ax.set_xticks([], minor = True)
plt.xlabel('ISO')
plt.ylabel('Relative RMSE × 100')
plt.grid(axis='y', which='major')
plt.grid(axis='y', which='minor', linestyle=(0, (5, 8)))
plt.legend(labels)
plt.savefig('comparison.pdf', bbox_inches='tight')
plt.show()
