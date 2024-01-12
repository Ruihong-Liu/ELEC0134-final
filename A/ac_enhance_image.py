"""
enhance the data expand the dataset to have better result
"""
from PIL import Image, ImageEnhance,ImageFilter
import random
import numpy as np
import matplotlib.pyplot as plt
# enhance datasets
def enhanced_data(train_images, train_labels):
    # enhance datasets
    def augment_image(image):
        # random flip horizontaly or vertically
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # random rotate between -30 to 30 degrees
        rotation_angle = random.randint(-30, 30)
        image = image.rotate(rotation_angle)

        # random change the brightness of the image
        enhancer = ImageEnhance.Brightness(image)
        brightness_factor = random.uniform(0.8, 1.2)
        image = enhancer.enhance(brightness_factor)

        # Apply Gaussian noise with a probability of 0.3
        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))
        
        return image

    # combine the data of enhaced and origion
    def augment_and_merge_dataset(images, labels, num_augmentations=5):
        #initialise the empty array to store the augmented image and the label of the image
        augmented_images = []
        augmented_labels = []
        #match the image and it's label
        for i in range(len(images)):
            original_image = Image.fromarray(images[i])
            augmented_images.append(images[i])
            augmented_labels.append(labels[i])

            # enhance the data and add them to the origional and form the new dataset
            for _ in range(num_augmentations):
                augmented_image = augment_image(original_image)
                augmented_images.append(np.array(augmented_image)) 
                augmented_labels.append(labels[i])

        return np.array(augmented_images), np.array(augmented_labels)
    augmented_train_images, augmented_train_labels = augment_and_merge_dataset(train_images, train_labels)
    #random plotting the image after enhanced
    num_samples = 5  # number of samples that needs to be show
    sample_indices = np.random.choice(augmented_train_images.shape[0], num_samples, replace=False)
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    for i, idx in enumerate(sample_indices):
        axes[i].imshow(augmented_train_images[idx])
        axes[i].set_title(f"Label: {augmented_train_labels[idx]}")
        axes[i].axis('off')
    # save the images
    plt.savefig(r"A\images\sample_augmented.png")
    # print the datasets check the size of the datasets
    print("\n")
    print("Augmented Train Images Shape:", augmented_train_images.shape)
    print("Augmented Train Labels Shape:", augmented_train_labels.shape)
    return augmented_train_images, augmented_train_labels