# -*- coding: utf-8 -*-
"""
Created on Tue Jan 3 17:30:49 2023

@author: Ajeeth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

#import the data

import pandas as pd
movie=pd.read_csv('my_movies.csv')
movie.head()

# Get list of categorical variables
s = (movie.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:",object_cols)

num_movie = movie.iloc[:,5:15]
num_movie.head()

#Apriori and rules

frequent_itemsets_ap = apriori(num_movie, min_support=0.15, use_colnames=True,verbose=1)
print(frequent_itemsets_ap.head())

frequent_itemsets_ap.sort_values("support", ascending = False).shape

rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=0.1)
print(rules_ap.head())


rules_ap['antecedents_'] = rules_ap['antecedents'].apply(lambda a: ','.join(list(a)))
rules_ap['consequents_'] = rules_ap['consequents'].apply(lambda a: ','.join(list(a)))
# Transform the DataFrame of rules into a matrix using the confidence metric
pivot = rules_ap[rules_ap['lhs items']>1].pivot(index = 'antecedents_', 
                    columns = 'consequents_', values= 'confidence')
# Generate a heatmap with annotations 
sns.heatmap(pivot, annot = True)
plt.title('Heat Map - For Confidence Metric')
plt.yticks(rotation=0)
plt.xticks(rotation=90)


















































































