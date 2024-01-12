import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
def train_resnet(train_images, train_labels,val_images, val_labels,test_images, test_labels):
    # 将标签转换为独热编码
    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)
    test_labels = to_categorical(test_labels)

    # 调整图像大小（ResNet50 默认输入尺寸至少为 32x32）
    train_images_resized = tf.image.resize(train_images, [32, 32])
    val_images_resized = tf.image.resize(val_images, [32, 32])
    test_images_resized = tf.image.resize(test_images, [32, 32])

    # 加载预训练的 ResNet50 模型
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # 冻结基础模型
    base_model.trainable = False

    # 添加自定义层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    num_classes = train_labels.shape[1]  # 根据您的数据集类别数
    predictions = Dense(num_classes, activation='softmax')(x)

    # 构建最终模型
    model = Model(inputs=base_model.input, outputs=predictions)

    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    history= model.fit(train_images_resized, train_labels, batch_size=32, epochs=10, validation_data=(val_images_resized, val_labels))

    # 评估模型
    test_loss, test_acc = model.evaluate(test_images_resized, test_labels)
    print("Test accuracy:", test_acc)
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

    plt.savefig("B\images\resnet train.png")
    return history