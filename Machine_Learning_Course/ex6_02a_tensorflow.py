#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#%%

def load():
    data = np.loadtxt("https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/data/data2Class.txt")
    #data=data[:100,:]
    X_data, y_data = data[:, :2], data[:, 2]
    
    
    X = tf.placeholder(tf.float32,shape=[None, 2])
    y = tf.placeholder('float')

    
    return X,y,X_data,y_data
#%%
def create_neural_net():
    X,y,X_data,y_data=load()
    print("################")
    print(X_data.shape)
    # relu_layer_operation = tf.layers.Dense(units=100,activation=tf.nn.leaky_relu,
    #     kernel_initializer=tf.initializers.random_uniform(-.1,.1),
    #     bias_initializer=tf.initializers.random_uniform(-1.,1.))
    # relu_layer_operation2 = tf.layers.Dense(units=100,activation=tf.nn.leaky_relu,
    #     kernel_initializer=tf.initializers.random_uniform(-.1,.1),
    #     bias_initializer=tf.initializers.random_uniform(-1.,1.))
    print(X)

    hidden1=tf.layers.dense(inputs=X,units=100,activation=tf.nn.leaky_relu,
        kernel_initializer=tf.initializers.random_uniform(-.1,.1),
        bias_initializer=tf.initializers.random_uniform(-1.,1.))
    print(hidden1)

    hidden2=tf.layers.dense(inputs=hidden1,units=100,activation=tf.nn.leaky_relu,
        kernel_initializer=tf.initializers.random_uniform(-.1,.1),
        bias_initializer=tf.initializers.random_uniform(-1.,1.))
    print(hidden2)

    linear_layer_operation = tf.layers.Dense(1,activation=None,
    kernel_initializer=tf.initializers.random_uniform(-.1,.1),
    bias_initializer=tf.initializers.random_uniform(-.01,.01))

    #loss=tf.Variable(0,dtype=tf.float32)
    # hidden1=relu_layer_operation(X)
    # print(hidden1)
    # #hidden1=tf.reshape(hidden1,[10000,2])
    # hidden2=relu_layer_operation(hidden1)
    model_output=linear_layer_operation(hidden2)
    print(model_output)

    loss = tf.reduce_mean(tf.losses.hinge_loss(logits=model_output, labels=y))
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-08).minimize(loss)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs_02', sess.graph)
        
        # for i in range(len(X_data)):  
        #     print("################")
        #     print(i)  
        init = tf.global_variables_initializer()
        sess.run(init)
        print(sess.run([optimizer,loss],feed_dict={X:X_data,y:y_data}))
            ##print(sess.run(loss,feed_dict={X:X_data[i,:].reshape((1,2)),y:y_data[i]}))
            
            #print(sess.run([optimizer,loss],feed_dict={X:X_data[i,:].reshape((1,2)),y:y_data[i]}))
        
        writer.close()
#%%
create_neural_net()