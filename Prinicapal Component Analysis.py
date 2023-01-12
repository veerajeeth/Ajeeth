c# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import the data
import pandas as pd
df = pd.read_csv("D:\\Data Science\\Assignments\\Principal Component Analysis\\wine.csv")
df
df.dtypes

df.info()

# Converting data to numpy array
df1=df.values
df1
    
# Normalizing the  numerical data

from sklearn.preprocessing import scale
Wine=scale(df1)
Wine

# Applying PCA Fit Transform to dataset

from sklearn.decomposition import PCA

pca = PCA()
pca_values = pca.fit_transform(Wine)
pca_values            
    
##Percentage of varaiance'

var=pca.explained_variance_ratio_ 
sum(pca.explained_variance_ratio_)       
        
##Graph

final_df=pd.concat([df['Type'],pd.DataFrame(pca_values[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)
final_df      
        
# Visualization of PCAs
import seaborn as sns
fig=plt.figure(figsize=(16,12))
sns.scatterplot(data=final_df);        
        
sns.scatterplot(data=final_df, x='PC1', y='PC2', hue='Type');
        
#Clustering

#Create dendograms

plt.figure(figsize=(10,8))
dendrogram=sch.dendrogram(sch.linkage(Wine,'complete'))        
        
# Create Clusters
clusters=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
clusters
AgglomerativeClustering(n_clusters=5)
y=pd.DataFrame(clusters.fit_predict(Wine),columns=['clustersid'])
y['clustersid'].value_counts()        
        
# Adding clusters to dataset
wine2=df.copy()
wine2['clustersid']=clusters.labels_
wine2        
        

#K Means

from sklearn.cluster import KMeans

# As we already have normalized data
# Use Elbow Graph to find optimum number of  clusters (K value) from K values range
# The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion WCSS 
# random state can be anything from 0 to 42, but the same number to be used everytime,so that the results don't change.        
        
        
# within-cluster sum-of-squares criterion 
wcss=[]
for i in range (1,6):
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(Wine)
    wcss.append(kmeans.inertia_)        
        
# Plot K values range vs WCSS to get Elbow graph for choosing K (no. of clusters)
plt.plot(range(1,6),wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS');        
        
# Cluster algorithm using K=3
clusters3=KMeans(5,random_state=40).fit(Wine)
clusters3
KMeans(n_clusters=6, random_state=40)
clusters3.labels_        

# Assign clusters to the data set
wine3=df.copy()
wine3['clusters3id']=clusters3.labels_
wine3

wine3['clusters3id'].value_counts()