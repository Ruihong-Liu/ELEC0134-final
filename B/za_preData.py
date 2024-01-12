"""
data pre-processing before train the neural network
"""
def dataPrepare():
    #load data from the file
    from aa_readfile import Dataread
    file_path = 'Datasets/pathmnist.npz'
    dataset=Dataread(file_path)

    #load each category as a variable
    from aa_readfile import category_Data
    train_images,train_labels,val_images,val_labels,test_images,test_labels=category_Data(dataset)

    # Random plot of the images
    from ab_sampleB import plot_sample
    plot_sample(train_images, train_labels)

    # Normalisation of the images
    from ac_Normalisation import NormalisationB
    train_images_normalized,val_images_normalized,test_images_normalized=NormalisationB(train_images,val_images,test_images)
    # resize the image from normalised data resize from 28-28 to 224-224
    
    return train_images,train_labels,val_images,val_labels,test_images,test_labels,train_images_normalized,val_images_normalized,test_images_normalized

    
    

