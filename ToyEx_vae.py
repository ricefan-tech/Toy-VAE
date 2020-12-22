# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:15:16 2020

@author: t656703
"""
import tensorflow as tf

import seaborn as sns
import scipy.stats as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
from vae_tf1 import VAE
import mgarch as mg
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

#generate 3-dimensional samples with desired correlation and marginal distribution
def getdata(dof=5, samp=10000, mean=[0.,2.,0.], final_cov=[1.,2.,1.], cov=[[1.,-0.8,0.],
                                                                           [-0.8,1.,0.5],
                                                                           [0.,0.5,1.]]):
 
    mvnorm=st.multivariate_normal(mean=[0.,0.,0.], cov=cov)
    x=mvnorm.rvs(samp, random_state=7)
    norm=st.norm()
    x_unif=norm.cdf(x)
    m0=st.norm(loc=mean[0], scale=final_cov[0])
    m1=st.norm(loc=mean[1], scale=final_cov[1])
    m2=st.t(df=dof, loc=mean[2], scale=final_cov[2])
    x0=m0.ppf(x_unif[:,0]).reshape(samp,1)
    x1=m1.ppf(x_unif[:,1]).reshape(samp,1)
    x2=m2.ppf(x_unif[:,2]).reshape(samp,1)
    return np.concatenate((x0,x1,x2), axis=1)

#plot histogram of data and correlation heatmap  
def plots(data, labels):
    axlabels=[]
    #histogram
    for i in range(len(labels)):
        plt.figure()
        h=sns.histplot(data=data[:,i],stat='probability',bins=80, linewidth=0.1)
        plt.xlabel(labels[i])
        plt.ylabel("Probability")
        plt.xlim(-8,8)
        plt.show()
        #plt.savefig(labels[i] + ".png")
        #plt.close()
        axlabels.append("Dataset "+str(i))
        
    #correlation matrix 
    p=pd.DataFrame(data)
    plt.figure()
    ax=sns.heatmap(p.corr(), annot=True, cbar=False, cmap="Blues", xticklabels=axlabels,yticklabels=axlabels, square=True)
    plt.show()
    #plt.savefig("correlation "+ labels[0])
    #plt.close()
    
#compute some summary statistics
def statis(data):
    if data.ndim!=3:
        a=pd.DataFrame(columns=["Mean", "Std. Dev.", "1st perc", "99th perc"], index=np.arange(data.shape[1]))
        #if its only real data and not samples
        a["Mean"]=np.mean(data, axis=0)
        a["Std. Dev."]=np.std(data, axis=0)
        a["1st perc"]=np.percentile(data, 0.01, axis=0)
        a["99th perc"]=np.percentile(data, 0.99, axis=0)
    else:
        a=pd.DataFrame(columns=["Mean","Mean Std", "Std. Dev.","Std. Std.", "1st perc","1st perc Std", "99th perc", "99th perc Std"], index=np.arange(data.shape[1]))
        #3d means it is concatenated along third dimension of simulation runs
        a["Mean"]=np.mean(data, axis=(0,2))
        a["Mean Std"]=np.std(np.mean(data,axis=0), axis=-1)
        a["Std. Dev."]=np.mean(np.std(data, axis=0), axis=-1)
        a["Std. Std."]=np.std(np.std(data,axis=0), axis=-1)
        a["1st perc"]=np.mean(np.percentile(data, 0.01, axis=0), axis=-1)
        a["1st perc Std"]=np.std(np.percentile(data, 0.01, axis=0), axis=-1)
        a["99th perc"]=np.mean(np.percentile(data, 0.99, axis=0), axis=-1)
        a["99th perc Std"]=np.std(np.percentile(data, 0.99, axis=0), axis=-1)
    return a

#QQ Plots of model generated data vs known theoretical distribution
def QQ(data,dist, dof=5, mean=[0.,2.,0.], cov=[1.,2.,1.]):
  
    for i in range(data.shape[1]):
        if dist[i] == "st.t":
            plt.figure()
            fig=sm.qqplot(data[:,i], eval(dist[i]), distargs=(dof,),loc=mean[i], scale=cov[i], line="45")
        else:
            plt.figure()            
            fig=sm.qqplot(data[:,i], eval(dist[i]),  loc=mean[i], scale=cov[i], line="45")


#%%    
if __name__ == '__main__':
    data=getdata()
    
    real_lab=["$\mathcal{N}$(0,1)", "$\mathcal{N}$(2,2)","$t_{5}$(0,1)"]
    scaler = MinMaxScaler(feature_range=(0.00001, 0.99999))
    data_scaled = scaler.fit_transform(data)
   
    #create and train VAE
    VAE=VAE(n_latent=2, n_hidden=100,alpha=0.5)
    VAE.train(data=data_scaled, n_epochs=1000, learning_rate=0.0001, show_progress=True)
    
    #%%
    #robustness test with statistics calculation/plotting 
    path_len=5000
    num_path=20
    res=np.zeros((path_len, data.shape[1],num_path))
    for i in range(num_path):
        res_norm=CVAE.generate(5000)
        res[:,:,i]=scaler.inverse_transform(res_norm)   
    stat_cvae=statis(res)
    res.sort(axis=0)
    mean_res=np.mean(res, axis=-1)
    dist=["st.norm", "st.norm", "st.t"]
    QQ(mean_res, dist)
   
    
