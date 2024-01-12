"""
main file for task A, call each .py file
"""
## first pre-process the image for later use
from za_preData import dataPrepare
train_images_normalized,augmented_train_images_normalized,val_images_normalized,test_images_normalized,train_labels,val_labels,test_labels,augmented_train_labels=dataPrepare()
##train CNN model with original data and plot the table
from ba_trainCNN import train_model_origional
history_ori=train_model_origional(train_images_normalized,val_images_normalized,test_images_normalized,train_labels,val_labels,test_labels)
json_file_ori="A/training_results_ori.json"
png_file_ori="A/images/table_ori.png"
from bb_table import table
table(history_ori,json_file_ori)
from bb_table import table_show
table_show(json_file_ori,png_file_ori)
##train CNN model with augmentation data and plot the table
from ba_trainCNN import train_model_augmention
history_aug=train_model_augmention(augmented_train_images_normalized,augmented_train_labels,val_images_normalized,test_images_normalized,val_labels,test_labels)
json_file_aug="A/training_results_aug.json"
png_file_aug="A/images/table_aug.png"
from bb_table import table
table(history_aug,json_file_aug)
from bb_table import table_show
table_show(json_file_aug,png_file_aug)