## Download the BelgiumTS dataset from: https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip (Training data) and https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip (Test data)
#%%
import os
import skimage
from skimage import transform
from skimage.color import rgb2gray
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D
import pandas as pd

#%%
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
    if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        #print(label_directory)
        file_names = [os.path.join(label_directory, f)
                    for f in os.listdir(label_directory)
                    if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            #print(d)
            labels.append(int(d))
    return np.array(images), np.array(labels)


def plot_data(signs, labels):
    for i in range(len(signs)):
        plt.subplot(4, len(signs)/4 + 1, i+1)
        plt.axis('off')
        plt.title("Label {0}".format(labels[i]))
        plt.imshow(signs[i])
        plt.subplots_adjust(wspace=0.5)
    plt.show()

#%%
train_images, train_labels = load_data("./Training")
test_images, test_labels=load_data('./Testing')
# display 30 random images
randind = np.random.randint(0, len(train_images), 30)
#plot_data(train_images[randind], train_labels[randind])


train_images = rgb2gray(np.array([transform.resize(image, (50, 50)) for image in train_images]))
test_images = rgb2gray(np.array([transform.resize(image, (50, 50)) for image in test_images]))
#%%
def create_neural_net_keras(X_train,y_train,X_test,y_test,batch_size,epochs=1):
    #graph=tf.Graph()
    #with graph.as_default():
    tensorboard_callback=tf.keras.callbacks.TensorBoard('logs_2b_tf_keras_2',write_graph=True,batch_size=batch_size) 
    model=Sequential()
    model.add(Conv2D(input_shape=(50,50,1),filters=64,kernel_size=[3,3],))
    model.add(MaxPool2D(pool_size=(2,2),strides=2))
    
    #model.add(Dense(units=128,activation='relu'))
    #print(Dense(units=128,activation='relu'))
    model.add(Conv2D(filters=64,kernel_size=[3,3],))
    model.add(MaxPool2D(pool_size=(2,2),strides=2))
    # model.add(Conv2D(filters=128,kernel_size=[3,3],))
    # model.add(MaxPool2D(pool_size=(2,2),strides=2))
    # model.add(Conv2D(filters=128,kernel_size=[3,3],))
    # model.add(MaxPool2D(pool_size=(2,2),strides=2))
    model.add(Flatten())
    model.add(Dense(units=128,activation='relu')) # units or filter are number of output neurons

    model.add(Dense(units=62,activation='softmax')) # 62 classes
    model.compile(optimizer='adam',loss='hinge', metrics=['accuracy'])
    
    model.fit(X_train, y_train, batch_size=batch_size,shuffle=True,epochs=epochs,callbacks=[tensorboard_callback])
    
    model.evaluate(X_test,y_test)
    return model
#%%
train_images=train_images.reshape((4575,50,50,1))
print(train_images.shape) 

print(train_labels)

train_labels=np.eye(62)[train_labels] #one-hot encoding
print(train_labels.shape)
test_images=test_images.reshape((2520,50,50,1))
print(test_images.shape)
test_labels=np.eye(62)[test_labels]#one-hot encoding
#print(test_labels.shape)
model=create_neural_net_keras(train_images,train_labels,test_images,test_labels,batch_size=183,epochs=50)


#%%
print(model.summary())

#tf.keras.utils.plot_model(model,'model_ex6_2b.png',show_shapes=True,show_layer_names=True)
# train_labels=ytrain
# dummy=np.eye(62)[train_labels]