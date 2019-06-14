#%%
#import and image paths
import matplotlib.pyplot as plt
import scipy as sp
import skimage 
import numpy as np
import os
from skimage.color import rgb2gray
import scipy.sparse.linalg as sla
data_directory='./yalefaces'

#%%

#Read and load images

def load_yale_faces(data_directory):
    
    images = [] 
    file_names=[]
    file_names = [os.path.join(data_directory, f)
                for f in os.listdir(data_directory) if not f.endswith('.txt')]
    
    print(f'file names ={file_names}')
    for f in file_names:

        images.append(skimage.data.imread(f))
            
    return np.array(images)
img=load_yale_faces(data_directory)
#%%
#Plot images
def plot_img(img=None):
    for i in range(len(img)):
        #print(i)
        plt.subplot(4, len(img)/4 + 1, i+1) #nrows,ncols,index
        plt.imshow(img[i])
        
#%%
#img1=img.reshape(165,77760)
#%%
#RGB2GRAY / Centering around mean
print(len(img))
X=rgb2gray(np.array(img))
print(X)

print(X.shape)
X_mean=np.mean(X,axis=0)
X_hat=X-np.ones((166,1))*X_mean.T
#%%
#PCA using SVD

num_eig_val=60
u,s,vt=sla.svds(X_hat,k=num_eig_val)
#eig_val,eig_vec=sla.eigs(X_hat)
#%%

#DR/Recon/Recon Error 
Z=np.matmul(X_hat,vt.T) #DR version of X
X_recon=np.ones((166,1))*X_mean.T + np.matmul(Z,vt)
recon_error=np.linalg.norm((X-X_recon),ord=2)


#%%
# print(os.listdir(data_directory))
    # for f in os.listdir(data_directory):
    #     if not f.endswith('.txt'):
    #         file_names.append(os.path.join(data_directory, f))