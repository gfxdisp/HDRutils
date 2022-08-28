from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(9, 3))

for i, dataset in enumerate(('fairchild', 'hdr4cv', 'linkoping')):
	data = np.genfromtxt(f'ratios_{dataset}.csv', delimiter=',').flatten()
	ax[i].hist(data[data!=0]*100, 50)
	ax[i].set_yticks([])
	ax[i].set_title(dataset)
	ax[i].set_xlabel('Relative error (in %)')
	print(f'Std {dataset}: {data.std()}')

ax[0].set_ylabel('Number of images')

plt.show()
# fig.tight_layout()
# fig.savefig('ratios.pdf')
