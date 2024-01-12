"""
random shows some image as an example
"""
import matplotlib.pyplot as plt
import random
# random select image and show 5 images
def display_random_images(images, labels, num_images=5):
    # random select
    random_indices = random.sample(range(len(images)), num_images)
    # image format
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 3))
    if num_images == 1:
        axes = [axes]

    # show each image that selected and save it for later use
    for i, index in enumerate(random_indices):
        ax = axes[i]
        ax.imshow(images[index], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Label: {labels[index]}')
    plt.suptitle("Ramdom selected 5 images from dataset",fontsize=16)
    plt.savefig("A\images\sample.png")
