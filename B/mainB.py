# load the data after pre-processed
from za_preData import dataPrepare
train_images,train_labels,val_images,val_labels,test_images,test_labels,train_images_normalized,val_images_normalized,test_images_normalized=dataPrepare()
# train the CNN model with 28*28 dataset
#from ba_TrainCNNB import train_model
#history1=train_model(train_images_normalized,val_images_normalized,test_images_normalized,train_labels,val_labels,test_labels)

from bb_train_resnet50 import train_resnet
history2=train_resnet(train_images_normalized,train_labels,val_images_normalized,val_labels,test_images_normalized,test_labels)