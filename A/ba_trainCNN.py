"""
for trainng CNN model
"""
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.regularizers import l2
def train_model_origional(train_images_normalized,val_images_normalized,test_images_normalized,train_labels,val_labels,test_labels):
     # L2 regulaisation
    l2_reg = 0.001
    #train the model CNN
    model = Sequential([
        Conv2D(5, kernel_size=3, activation='relu', input_shape=(train_images_normalized.shape[1], train_images_normalized.shape[2], 1), kernel_regularizer=l2(l2_reg)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(9, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_reg)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(5, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_reg)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(0.5),  # Dropout reduce overfit
        Dense(1, activation='sigmoid')
        ])
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # train the model
    history = model.fit(train_images_normalized, train_labels, validation_data=(val_images_normalized, val_labels), epochs=10, batch_size=32)

    # test the accuary
    test_loss, test_accuracy = model.evaluate(test_images_normalized, test_labels)
    print("Test accuracy:", test_accuracy)
    print("Test loss:", test_loss)
    # plot accuracy and loss of the model
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.suptitle("The result of the model without expanded dataset",fontsize=16)
    plt.savefig("A/images/Training and testing origional data.png")
    return history

# training with enhanced data
def train_model_augmention(augmented_train_images_normalized,augmented_train_labels,val_images_normalized,test_images_normalized,val_labels,test_labels):
    
    # L2 regulaisation
    l2_reg = 0.001
    #train the model CNN
    model = Sequential([
        Conv2D(5, kernel_size=3, activation='relu', input_shape=(augmented_train_images_normalized.shape[1], augmented_train_images_normalized.shape[2], 1), kernel_regularizer=l2(l2_reg)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(9, kernel_size=3, activation='sigmoid', kernel_regularizer=l2(l2_reg)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(5, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_reg)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(0.5),  # Dropout reduce overfit
        Dense(1, activation='sigmoid')
        ])
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # train the model
    history = model.fit(augmented_train_images_normalized, augmented_train_labels, 
                        validation_data=(val_images_normalized, val_labels), 
                        epochs=13, batch_size=32)

    # test the accuary
    test_loss, test_accuracy = model.evaluate(test_images_normalized, test_labels)
    print("Test accuracy:", test_accuracy)
    print("Test loss:", test_loss)
    # plot accuracy and loss of the model
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.suptitle("The result of the model with expanded dataset",fontsize=16)
    plt.savefig("A\images\Training and testing enhanced data.png")
    return history