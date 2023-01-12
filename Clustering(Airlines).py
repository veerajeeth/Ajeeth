c# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


# Importing the data
import pandas as pd
df = pd.read_csv("EastWestAirlines.csv")
df

df.shape
df.head()
df.dtypes

## drop the variable

df1=df.drop(["ID"],axis=1)
df1

#Noramlize the numeric data

from sklearn.preprocessing import normalize
df1_norm=pd.DataFrame(normalize(df1),columns=df1.columns)
df1_norm

#Create dendograms

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(df1_norm, method='complete')) 

##create clustering

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='complete')
Y = cluster.fit_predict(df1_norm)
Y

Y = pd.DataFrame(Y)
Y[0].value_counts()

plt.figure(figsize=(10, 7))  
plt.scatter(df1['Qual_miles'],df1['Balance'], c=cluster.labels_)    
    

# K-Means clustering

from sklearn.cluster import KMeans
Kmeans = KMeans(n_clusters=5,n_init=20)
Kmeans.fit(df1_norm)
Y = Kmeans.predict(df1_norm)

Y = pd.DataFrame(Y)
Y[0].value_counts()

plt.figure(figsize=(10, 7))  
plt.scatter(df1_norm['Qual_miles'],df1_norm['Balance'],c=Kmeans.labels_,)  

Kmeans.inertia_    
    
# Getting the cluster centers
C = Kmeans.cluster_centers_

l1 = []
for i in range(1,11):
    Kmeans = KMeans(n_clusters=i,n_init=20)
    Kmeans.fit(df1_norm)
    l1.append(Kmeans.inertia_)
    
print(l1)

pd.DataFrame(range(1,11))        
pd.DataFrame(l1)
    
pd.concat([pd.DataFrame(range(1,11)),pd.DataFrame(l1)], axis=1)

import matplotlib.pyplot as plt
plt.scatter(range(1,11),l1)
plt.show()

# DBSCAN Clustering
from sklearn.cluster import DBSCAN

df = df.iloc[:,1:11]
df.values

# Normalize heterogenous numerical data using standard scalar fit transform to dataset
from sklearn.preprocessing import StandardScaler
SS = StandardScaler().fit(df.values)
x = SS.transform(df.values)

DBS = DBSCAN(eps=2,min_samples=4)
DBS.fit(x)

# Noisy samples are given the label -1.
DBS.labels_

# Adding clusters to dataset
c1 = pd.DataFrame(DBS.labels_,columns=['cluster'])
c1

pd.concat([df,c1],axis=1)



    
    
    
    
    
    
        
    
        
    


            
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        