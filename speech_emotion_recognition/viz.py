import matplotlib.pyplot as plt
import numpy as np

def plot_unique_series_emotions(x, y):
    u, indices = np.unique(y, return_index=True)
    emotion_samples = x[indices]

    fig, axs = plt.subplots(2, 4, figsize=(30, 10))
    for i, sample_index in enumerate(indices):
        row = i % 2
        axs.flatten()[i].plot(x[sample_index])
        axs.flatten()[i].set_title(y[sample_index])
    plt.show()