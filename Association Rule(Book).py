# -*- coding: utf-8 -*-
"""
Created on Tue Jan 3 15:29:18 2023

@author: Ajeeth
"""

#Install apyori and mlxtend

pip install apyori

pip install mlxtend

#Import the data

import pandas as pd
df=pd.read_csv("book.csv")
df

df.head()
df.dtypes

#counting the values

df["ChildBks"].value_counts()
df["YouthBks"].value_counts()
df["CookBks"].value_counts()
df["DoItYBks"].value_counts()
df["RefBks"].value_counts()

#Apriori Algorithm and rules

from mlxtend.frequent_patterns import apriori,association_rules

frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets
frequent_itemsets.shape

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules.shape
list(rules)

rules.sort_values('lift',ascending = False)

rules.sort_values('lift',ascending = False)[0:20]

rules[rules.lift>1]

rules[['support','confidence']].hist()

rules[['support','confidence','lift']].hist()

#Scatter plot between support and confidence

import matplotlib.pyplot as plt
plt.scatter(rules['support'], rules['confidence'])
plt.show()

import seaborn as sns
sns.scatterplot('support', 'confidence', data=rules, hue='antecedents')

plt.show()


# Generate a heatmap with annotations 

import seaborn as sns
sns.heatmap(df,annot = True)
plt.title('Heat Map - For Confidence Metric')
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()


pip install nsepython

from nsepython import *
print(indices)

oi_data, ltp, crontime = oi_chain_builder("RELIANCE","latest","full")
print(oi_data)
print(ltp)
print(crontime)










































