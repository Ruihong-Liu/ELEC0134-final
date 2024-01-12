import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
def CNN_resnet(train_images_normalized,train_labels,val_images_normalized,val_labels,test_images_normalized,test_labels):
    # recategory the labels for resnet
    train_labels_cat = to_categorical(train_labels, num_classes=9)
    val_labels_cat = to_categorical(val_labels, num_classes=9)
    test_labels_cat = to_categorical(test_labels, num_classes=9)
    # ensure the label is float 32 format
    train_images_normalized = train_images_normalized.astype('float32')
    val_images_normalized = val_images_normalized.astype('float32')
    test_images_normalized = test_images_normalized.astype('float32')

    # pre- process the images to satify resnet requirment
    train_images_prep = preprocess_input(train_images_normalized)
    val_images_prep = preprocess_input(val_images_normalized)
    test_images_prep = preprocess_input(test_images_normalized)
    # load resnet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(28, 28, 3)))

    # change the output layer to satisify task
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(9, activation='softmax')(x)

    # comple model which is going to be used
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    history = model.fit(train_images_prep, train_labels_cat, batch_size=32, epochs=10, 
                        validation_data=(val_images_prep, val_labels_cat))
    # testing
    test_loss, test_accuracy = model.evaluate( test_images_prep, test_labels_cat)

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    # plot loss and accuracy
    plt.figure(figsize=(12, 4))

    # accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.suptitle("The result of resnet50",fontsize=16)
    plt.savefig("B\images\Training and testing resnet50.png")
    return history
