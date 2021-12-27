import numpy as np
from matplotlib import pyplot as plt

history = np.load("history.npy")
#plt.plot(history)
#plt.show()

population = np.load("population.npy")
fitness = np.load("fitness.npy")

print(population[:10])
