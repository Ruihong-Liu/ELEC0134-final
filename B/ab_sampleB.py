"""
random plotting of sample images printing 9 each categary has onw
"""
import matplotlib.pyplot as plt
import numpy as np
# plot random sample
def plot_sample(images, labels):
    #make sure each lable as one image
    unique_labels = np.unique(labels)
    plt.figure(figsize=(15, 15))
    #random select and plot
    for i, label in enumerate(unique_labels):
        idxs = np.where(labels == label)[0]
        random_idx = np.random.choice(idxs)

        plt.subplot(3, 3, i + 1)
        plt.imshow(images[random_idx])
        plt.title(f'Label: {label}')
        plt.axis('off')
    
    plt.suptitle('Random Sample for each lable')
    plt.savefig("B/images/SampleB.png")
