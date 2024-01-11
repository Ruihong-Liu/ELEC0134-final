import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

def train_model_resize(train_images_resized, val_images_resized, test_images_resized,train_labels,val_labels,test_labels):
    # number of categary
    num_classes = 9 
    # input image size
    input_shape = (224, 224, 3)  

    # create the model
    model = Sequential([
        
        Conv2D(5, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(5, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(5, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(5, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        # softmax for multi
        Dense(num_classes, activation='softmax')
    ])

    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    history = model.fit(train_images_resized, train_labels,
                    validation_data=(val_images_resized, val_labels),
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
    val_loss, val_accuracy = model.evaluate(val_images_resized, val_labels)
    test_loss, test_accuracy = model.evaluate(test_images_resized, test_labels)

    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    return history