"""
enhance the data
"""
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import random
import numpy as np
# enhance datasets
def enhanced_data(train_images, train_labels, num_augmentations=5):
    # enhance datasets
    def augment_image(image):
        # random flip left or right
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
        augmented_images = []
        augmented_labels = []

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
    # recall the function above
    augmented_train_images, augmented_train_labels = augment_and_merge_dataset(train_images, train_labels, num_augmentations=5)
    # print the datasets
    print("\n")
    print("Augmented Train Images Shape:", augmented_train_images.shape)
    print("Augmented Train Labels Shape:", augmented_train_labels.shape)
    return augmented_train_images, augmented_train_labels