#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#%%
def load():
    data = np.loadtxt("https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/data/data2Class.txt")
    data=data[:100,:]
    X_data, y_data = data[:, :2], data[:, 2]
    betas = np.zeros(shape=(3))
    def prepend_ones(X):
        '''prepends ones vector to X'''

        return np.column_stack([np.ones(X.shape[0]) , X])
    X_data=prepend_ones(X_data)
    X = tf.placeholder(tf.float32,shape=(100, 3))
    y = tf.placeholder(tf.float32,shape=(100))
    beta = tf.placeholder(tf.float32,shape=(3))
    print(X.shape)
    print(y.shape)
    print(beta.shape)
    return X,y,beta,betas,X_data,y_data


#%%
def compute_nll(X,y,beta,betas,X_data,y_data):
    ''' compute neg. log likelihood for logistic function'''
    print(X_data)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs', sess.graph)
        
        z = tf.tensordot(X,beta,axes=1)
        print(z.shape)
        p = tf.math.sigmoid(z)
        print(p.shape)
        loss=tf.Variable(0,dtype=tf.float32)
        #loss =-(y*tf.math.log(p)+(1-y)*(1-tf.math.log(p)))
        loss=-(tf.tensordot(y,tf.math.log(p),axes=1)+tf.tensordot((1-y),(tf.math.log(1-p)),axes=1))
        loss=tf.math.reduce_sum(loss)
        print(sess.run(z,feed_dict={X:X_data,y:y_data,beta:betas}))
        print(sess.run(p,feed_dict={X:X_data,y:y_data,beta:betas}))
        print(sess.run(loss,feed_dict={X:X_data,y:y_data,beta:betas}))
        
        

    return loss
    # for i in range(X.shape[0]):
    #     #loss=tf.math.subtract(loss,(y[i]*tf.math.log(p[i])+(1-y[i])*(1-tf.math.log(p[i]))))
    #     loss =-(y[i]*tf.math.log(p[i])+(1-y[i])*(1-tf.math.log(p[i])))
    #     #tf.assign_add(loss,l)
    #     print(sess.run(loss,feed_dict={X:X_data,y:y_data,beta:betas}))
    
#%%
def compute_gradients_hessians():
    X,y,beta,betas,X_data,y_data=load()
    print(X_data)
    loss=compute_nll(X,y,beta,betas,X_data,y_data)  
    print("Gradients & Hessians tensorflow")
    with tf.Session() as sess:
        g=tf.gradients(loss,beta)
        h=tf.hessians(loss,beta)
        print(sess.run(g,feed_dict={X:X_data,y:y_data,beta:betas}))
        print(sess.run(h,feed_dict={X:X_data,y:y_data,beta:betas}))
#%%
def numpy_equations():
    X,y,beta,betas,X_data,y_data=load()
    p = 1. / (1. + np.exp(-np.dot(X_data, betas)))
    print(p)
    #print(y_data * np.log(p) + ((1. - y_data) * np.log(1.-p)))
    L = -np.sum(y_data * np.log(p) + ((1. - y_data) * np.log(1.-p)))
    dL = np.dot(X_data.T, p - y_data)
    W = np.identity(X_data.shape[0]) * p * (1. - p)
    ddL = np.dot(X_data.T, np.dot(W, X_data))
    print("#################")
    print("Loss")
    print(L)
    print("Gradients & Hessians")
    print(dL)
    print(ddL)
    return L, dL, ddL
#%%
numpy_equations()
print("####################")
compute_gradients_hessians()
