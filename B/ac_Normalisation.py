"""
Normalisation
"""
def NormalisationB(train_images,val_images,test_images):
    train_images_normalized = train_images.astype('float32') / 255.0
    val_images_normalized = val_images.astype('float32') / 255.0
    test_images_normalized = test_images.astype('float32') / 255.0
    print("range of the data after normlisation：")
    print(f"training set：{train_images_normalized.min(), train_images_normalized.max()}")
    print(f"validation set：{val_images_normalized.min(), val_images_normalized.max()}")
    print(f"testing set：{test_images_normalized.min(), test_images_normalized.max()}")
    return train_images_normalized,val_images_normalized,test_images_normalized