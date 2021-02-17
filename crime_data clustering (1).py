#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
df=pd.read_csv("crime_data (1).csv")
df


# In[6]:


def norm_func(i):
    x=(i-i.mean()/i.std())
    return x 


# In[7]:


df_norm = norm_func(df.iloc[:,1:])
df_norm


# In[8]:



from sklearn.preprocessing import MinMaxScaler
trans = MinMaxScaler()
data = pd.DataFrame(trans.fit_transform(df.iloc[:,1:]))
data 


# In[9]:


z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title(' Clustering Dendrogram')
sch.dendrogram(
    z,
    #leaf_rotation=0.,  # rotates the x axis labels
    #leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[10]:


from sklearn.cluster import AgglomerativeClustering
model=AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="complete").fit(df_norm)
clusters_labels=pd.Series(model.labels_)
df["Clusters"]=clusters_labels
df


# In[15]:


df.iloc[:,1:].groupby(df.clust).mean()


# In[16]:


df.iloc[:,1:].groupby(df.clust).median()


# In[11]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
trans = StandardScaler()
data = pd.DataFrame(trans.fit_transform(df.iloc[:,1:]))
data 


# In[12]:



from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 
#p = np.array(df_norm) # converting into numpy array format 
z = linkage(data, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
sch.dendrogram(
    z,
    #leaf_rotation=0.,  # rotates the x axis labels
    #leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[13]:


from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=4, linkage='complete',affinity = "euclidean").fit(data) 

cluster_labels=pd.Series(h_complete.labels_)
cluster_labels
df['clust']=cluster_labels # creating a  new column and assigning it to new column 
df 


# In[17]:


data1=df[(df.clust==2)]
data1


# In[ ]:


#we conclude that there are four clusters  and the average crime is 3.772 which is in cluster 2(data1).

