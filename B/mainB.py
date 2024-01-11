# load the data after pre-processed
from B.za_preData import dataPrepare
train_images,train_labels,val_images,val_labels,test_images,test_labels,train_images_normalized,val_images_normalized,test_images_normalized,train_images_resized, val_images_resized, test_images_resized=dataPrepare()
# train the CNN model with 28*28 dataset
from B.ba_TrainCNNB import train_model
history1=train_model(train_images_normalized,val_images_normalized,test_images_normalized,train_labels,val_labels,test_labels)
# train the CNN model with 224*224 dataset
from B.bb_CNNresize import train_model_resize
history2=train_model_resize(train_images_resized, val_images_resized, test_images_resized,train_labels,val_labels,test_labels)
