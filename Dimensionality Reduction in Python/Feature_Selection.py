#!/usr/bin/env python
# coding: utf-8

# In[1]:


#the curse of dim. ,, models tend to overfit high dim. data so how to detect low quality features?

from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def drop_var_threshold(df,threshold = 0.05):
    """"
    dropping features that has variance < threshold
    
    """"
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(df/df.mean())
    mask = sel.get_support()
    reduced_df = df.loc[:,mask]
    return reduced_df


# In[2]:


def drop_na_threshold(df,threshold = 0.3):
    """"
    dropping features that has nan values > threshold
    
    """"
    mask =((df.isna().sum())/len(df)) < 0.3
    reduced_df = df.loc[:,mask]
    return reduced_df


# In[3]:


def corr_heatmap(df):
    """"
    Plotting heatmap of corr. coefs without displaying redundant coefs
    
    """"
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype = bool))
    cmap = sns.diverging_palette(h_neg=10,h_pos=240,as_cmap=True)
    sns.heatmap(df.corr(),mask=mask,center=0,cmap=cmap,linewidths=1,annot=True,fmt='.2f')
    plt.show()


# In[4]:


def drop_corr(df,threshold):
    """"
    dropping one of two features that has corr value > threshold
    
    """"
    corr_df = df.corr().abs()
    mask = np.triu(np.ones_like(corr_df,dtype=bool))
    tri_df = corr_df.mask(mask)
    to_drop = [col for col in tri_df.columns if any(tri_df[col]>threshold)]
    reduced_df = df.drop(to_drop,axis=1)
    return reduced_df


# In[ ]:




