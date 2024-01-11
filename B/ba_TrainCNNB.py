import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, MaxPooling2D

def train_model( train_images_normalized,val_images_normalized,test_images_normalized,train_labels,val_labels,test_labels):
    # Number of classes in your dataset
    num_classes = 9  # Replace with the actual number of classes
    train_labels= to_categorical(train_labels)
    val_labels= to_categorical(val_labels)
    test_labels= to_categorical(test_labels)
    # Building the CNN model
    model = Sequential([
    # First layer with 4 filters
    Conv2D(4, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Second layer with 9 filters
    Conv2D(9, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Third layer with 4 filters
    Conv2D(4, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Flattening to a single vector
    Flatten(),

    # Output layer with softmax activation
    Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images_normalized, train_labels,
                    validation_data=(val_images_normalized, val_labels),
                    epochs=10,  # 
                    batch_size=32)

    # 绘制准确率和损失图
    plt.figure(figsize=(12, 4))

    # 准确率图
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 损失图
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    # 评估模型性能
    # test_images_normalized, test_labels
    val_loss, val_accuracy = model.evaluate(val_images_normalized, val_labels)
    test_loss, test_accuracy = model.evaluate(test_images_normalized, test_labels)

    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    return history