import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.regularizers import l2
def train_resnet(train_images, train_labels,val_images, val_labels,test_images, test_labels):
    # 构建 ResNet50 模型
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(28, 28, 3)))

    # 添加自定义层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)  # L2 正则化
    predictions = Dense(train_labels.max() + 1, activation='softmax')(x)

    # 构建最终模型
    model = Model(inputs=base_model.input, outputs=predictions)

    # 冻结 ResNet50 的基础层
    for layer in base_model.layers:
        layer.trainable = False

    # 编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit(train_images, train_labels, batch_size=32, epochs=10, validation_data=(val_images, val_labels))

    # 评估模型
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    # 如果需要，可以对顶层进行微调
    for layer in model.layers[:143]:
        layer.trainable = False
    for layer in model.layers[143:]:
        layer.trainable = True

    # 重新编译模型（微调时）
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # 低学习率
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # 微调模型
    history=model.fit(train_images, train_labels, batch_size=32, epochs=5, validation_data=(val_images, val_labels))

    return history
