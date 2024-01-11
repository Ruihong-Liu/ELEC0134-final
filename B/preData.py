"""
data pre-processing before train the neural network
"""
#load data from the file
from readfile import Dataread
file_path = 'Datasets/pathmnist.npz'
dataset=Dataread(file_path)

#load each category as a variable
from readfile import category_Data
train_images,train_labels,val_images,val_labels,test_images,test_labels=category_Data(dataset)

# Random plot of the images
from sampleB import plot_sample
plot_sample(train_images, train_labels)

# Normalisation of the images
from Normalisation import NormalisationB
train_images_normalized,val_images_normalized,test_images_normalized=NormalisationB(train_images,val_images,test_images)

