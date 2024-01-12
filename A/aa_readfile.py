import numpy as np
def Dataread(file_path):
    # load data from file
    dataset = np.load(file_path)
    #check the categries of the dataset
    dataset.files
    return dataset
def category_Data(dataset):
    #load each category as a variable
    #train data
    train_images = dataset['train_images']
    train_labels = dataset['train_labels']
    #Validation data
    val_images = dataset['val_images']
    val_labels = dataset['val_labels']
    #test data
    test_images = dataset['test_images']
    test_labels = dataset['test_labels']

    # print the varibles
    #train
    print("Train Images Shape:", train_images.shape)
    print("Train Labels Shape:", train_labels.shape)
    print("\n")
    #Validation
    print("Validation Image Shape:", val_images.shape)
    print("Validation Labels Shape:", val_labels.shape)
    print("\n")
    #test
    print("Test Images Shape:", test_images.shape)
    print("Test Labels Shape:", test_labels.shape)
    print("\n")

    # close the file
    dataset.close()
    # return the function value for usage
    return  train_images,train_labels,val_images,val_labels,test_images,test_labels