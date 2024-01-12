"""
normalise data
"""
def normalisation(train_images,val_images,test_images,augmented_train_images):
    # normalise the data (Origional)
    train_images_normalized = train_images / 255.0
     # normalise the data (augmented)
    augmented_train_images_normalized = augmented_train_images / 255.0
    # normalise the data (validation)
    val_images_normalized = val_images / 255.0
    # normalise the data (test)
    test_images_normalized = test_images / 255.0
   
    # check the normalise result if it is between 0 and 1
    print("\n")
    print("Train Images (Origional)- Min:", train_images_normalized.min(), "Max:", train_images_normalized.max())
    print("Train Images (augmented)- Min:", augmented_train_images_normalized.min(), "Max:", augmented_train_images_normalized.max())
    print("validation Images- Min:", val_images_normalized.min(), "Max:", val_images_normalized.max())
    print("Test Images - Min:", test_images_normalized.min(), "Max:", test_images_normalized.max())
    # return the data after normalised
    return train_images_normalized,augmented_train_images_normalized,val_images_normalized,test_images_normalized