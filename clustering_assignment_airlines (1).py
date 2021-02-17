#!/usr/bin/env python
# coding: utf-8

# In[4]:


#HIERARCHIAL#
import numpy as np
import pandas as pd
import matplotlib.pylab as plt 
df = pd.read_csv("EastWestAirlines.csv")
df


# In[5]:


#Normalizing the data#
def norm_func(i):
    x=(i-i.min()/i.std())
    return (x)


# In[6]:


df_norm=norm_func(df.iloc[:,1:])
df_norm


# In[7]:


from sklearn.preprocessing import MinMaxScaler
trans = MinMaxScaler()
data = pd.DataFrame(trans.fit_transform(data.iloc[:,1:]))
data


# In[116]:


from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 
#p = np.array(df_norm) # converting into numpy array format 
z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(25, 50))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(
    z,
    #leaf_rotation=0.,  # rotates the x axis labels
    #leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[123]:


from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=7, linkage='complete',affinity = "euclidean").fit(df_norm) 

cluster_labels=pd.Series(h_complete.labels_)
cluster_labels
df['clust']=cluster_labels # creating a  new column and assigning it to new column 
df


# In[124]:


df.iloc[:,1:].groupby(df.clust).mean()


# In[125]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data1 = pd.DataFrame(scaler.fit_transform(df.iloc[:,1:]))
data1


# In[129]:


from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 
#p = np.array(df_norm) # converting into numpy array format 
z = linkage(data1, method="complete",metric="euclidean")
plt.figure(figsize=(25, 50))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(
    z,
    #leaf_rotation=0.,  # rotates the x axis labels
    #leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[130]:


from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=7, linkage='complete',affinity = "euclidean").fit(data) 

cluster_labels=pd.Series(h_complete.labels_)
cluster_labels
df['clust']=cluster_labels # creating a  new column and assigning it to new column 
df


# In[127]:


#KMEANS CLUSTERING#


# In[128]:


import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
import numpy as np


# In[19]:


X = np.random.uniform(0,1,1000)
Y = np.random.uniform(0,1,1000)
X


# In[20]:


df_xy =pd.DataFrame(columns=["X","Y"])
df_xy
df_xy.X = X
df_xy.Y = Y
df_xy
df_xy.plot(x="X",y = "Y",kind="scatter")
model1 = KMeans(n_clusters=5).fit(df_xy)


# In[21]:


df_xy.plot(x="X",y = "Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm_r)


# In[22]:


dt = pd.read_csv("EastWestAirlines.csv")
dt


# In[23]:


def norm_func(i):
    x = (i-i.min()) / (i.max() - i.min())
    return (x)


# In[26]:


dt_norm = norm_func(dt.iloc[:,1:])
dt_norm


# In[38]:


dt_norm.head(10)  # Top 10 rows


# In[39]:


from sklearn.cluster import KMeans
fig = plt.figure(figsize=(15, 12))
WCSS = []
for i in range(1, 15):
    clf = KMeans(n_clusters=i)
    clf.fit(dt_norm)
    WCSS.append(clf.inertia_) # inertia is another name for WCSS
plt.plot(range(1, 15), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()


# In[117]:


clf = KMeans(n_clusters=7)
y_kmeans = clf.fit_predict(df_norm)


# In[118]:


y_kmeans
#clf.cluster_centers_
clf.labels_


# In[119]:


md=pd.Series(y_kmeans)  # converting numpy array into pandas series object 
dt['clust']=md # creating a  new column and assigning it to new column 
dt


# In[120]:


dt.iloc[:,1:7].groupby(dt.clust).mean()


# In[121]:


dt.plot(x="Balance",y ="Bonus_miles",c=clf.labels_,kind="scatter",s=25 ,cmap=plt.cm.coolwarm) 
plt.title('Clusters using KMeans')


# In[122]:


clf.inertia_


# In[104]:


WCSS


# In[ ]:


#from Kmeans clustering the optimal number of clusters are 7 and from hierarchial clustering the average balance is at 5th cluster


# In[ ]:




