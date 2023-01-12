# -*- coding: utf-8 -*-
"""
Created on Wed jan 4 12:05:55 2023

@author: 10786
"""


#Import the data

import pandas as pd
df = pd.read_csv("book.csv", encoding = 'latin1')
df

df.dtypes

df.sort_values('User_ID')

# Get information of the datasets
df.info()
print('The shape of our data is:', df.shape)
df.isnull().any()

df.sort_values('User_ID')

# Number of unique users in the dataset
len(df)
len(df.User_ID.unique())


#EDA

df['Book_Rating'].value_counts()

# Histogram
df['Book_Rating'].hist()

len(df.Book_Title.unique())
df.Book_Title.value_counts()
t1 = df.Book_Title.value_counts()

# Bar graph
t1.plot(kind='bar')

# Use Pivot Table to reshape the data
user_df = df.pivot_table(index ='User_ID', columns='Book_Title', values='Book_Rating')

print(user_df)

# Number of # Impute those NaNs with 0 values 
user_df.fillna(0, inplace=True)
user_df


#Calculate cosine

from sklearn.metrics import pairwise_distances
user_sim=1-pairwise_distances(user_df.values,metric='cosine')
user_sim

#convert the results into data frame

user_sim_df=pd.DataFrame(user_sim)
user_sim_df


#Set the index and column names to User ID

user_sim_df.index=df.User_ID.unique()
user_sim_df.columns=df.User_ID.unique()

#Fill the diagonal values with 0

import numpy as np

np.fill_diagonal(user_sim,0)

user_sim_df.iloc[0:7,0:7]


# To save your cosin calcutaion file
user_sim_df.to_csv("cosin_calc.csv")

# Most Similar Users
user_sim_df.max()
user_sim_df.idxmax(axis=1)[0:10]


df[(df['User_ID'] == 276729) | (df['User_ID'] == 276726)]

df[(df['User_ID'] == 276736) | (df['User_ID'] == 276726)]

df[(df['User_ID'] == 276754) | (df['User_ID'] == 276726)]












