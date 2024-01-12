import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
#function to train the model
def train_model(train_images_normalized, val_images_normalized, test_images_normalized, train_labels, val_labels, test_labels):
    # change the label to one hot code
    train_labels_cat = to_categorical(train_labels, num_classes=9)
    val_labels_cat = to_categorical(val_labels, num_classes=9)
    test_labels_cat = to_categorical(test_labels, num_classes=9)
    # creat the model
    model = Sequential([
        #hidden layer one with input layer of 28*28*3
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 3), kernel_regularizer=l2(0.001)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        #hidden layer 2
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        #hidden layer 3
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        #hidden layer 4
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        #output
        Dense(9, activation='softmax')

    ])

    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    history = model.fit(train_images_normalized, train_labels_cat, batch_size=32, epochs=10, 
                        validation_data=(val_images_normalized, val_labels_cat))

    # testing 
    test_loss, test_accuracy = model.evaluate( test_images_normalized, test_labels_cat)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    # plotting the accuracy and loss graph
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

    plt.suptitle("The result of the self-designed model",fontsize=16)
    plt.savefig("B\images\Training and testing.png")
    return history