"""

"""
import matplotlib.pyplot as plt
import numpy as np
import random
# plot random sample
def plot_sample(images, labels):
    unique_labels = np.unique(labels)
    plt.figure(figsize=(15, 15))

    for i, label in enumerate(unique_labels):
        idxs = np.where(labels == label)[0]
        random_idx = np.random.choice(idxs)

        plt.subplot(3, 3, i + 1)
        plt.imshow(images[random_idx])
        plt.title(f'Label: {label}')
        plt.axis('off')

    plt.suptitle('Random Sample for each lable')
    plt.savefig("images/SampleB.png")
