

import numpy as np
import pandas as pd
df=pd.read_csv("50_Startups.csv")
df


# correlation 
df.corr()


#split the Variables in  X and Y's

# model 1
X = df[["R&D Spend"]] # R2: 0.947, RMSE: 9226.101

# Model 2
X = df[["R&D Spend","Administration"]] # R2: 0.948, RMSE: 9115.198

# Model 3
X = df[["Marketing Spend"]] # R2: 0.559, RMSE: 26492.829

# Model 4
X = df[["Marketing Spend","Administration"]] # R2: 0.610, RMSE: 24927.067

# Model 5
X = df[["R&D Spend","Marketing Spend","Administration"]] # R2: 0.951, RMSE: 8855.344

# Target
Y = df["Profit"]

# scatter plot between each x and Y  
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)
   
#======================================
import statsmodels.api as sma

X_new = sma.add_constant(X)
lmreg = sma.OLS(Y,X_new).fit()
lmreg.summary()


# Residual Analysis

import matplotlib.pyplot as plt
import statsmodels.api as sm
qqplot=sm.qqplot(lmreg.resid,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


lmreg.resid.hist()
lmreg.resid
list(np.where(lmreg.resid>10))

#model validation




from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test = train_test_split(X,Y , test_size=0.3,random_state=(42))

#=================================================================
# step5: model fitting
from sklearn.linear_model import LinearRegression

LR = LinearRegression()
LR.fit(X_train,Y_train)
Y_pred_train = LR.predict(X_train)
Y_pred_test = LR.predict(X_test)

#=================================================================
# step6: metrics
from sklearn.metrics import mean_squared_error
mse1= mean_squared_error(Y_train,Y_pred_train)
RMSE1 = np.sqrt(mse1)
print("Training error: ", RMSE1.round(2))

mse2= mean_squared_error(Y_test,Y_pred_test)
RMSE2 = np.sqrt(mse2)
print("Test error: ", RMSE2.round(2))

#model validation by r square 
Training_error = []
Test_error = []

for i in range(1,500):
    X_train, X_test,Y_train, Y_test = train_test_split(X,Y , test_size=0.3,random_state=(i))
    LR.fit(X_train,Y_train)
    Y_pred_train = LR.predict(X_train)
    Y_pred_test = LR.predict(X_test)
    Training_error.append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)).round(2))
    Test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)).round(2))
    

print(Training_error)
print(Test_error)
    
print("validationset approach for Traning: ",np.mean(Training_error).round(2))    
print("validationset approach for test: ",np.mean(Test_error).round(2))

# step 7 model deletion 
# Model Deletion Diagnostics
## Detecting Influencers/Outliers

## Cook’s Distance

lmreg_influence = lmreg.get_influence()
(cooks, pvalue) = lmreg_influence.cooks_distance

cooks = pd.DataFrame(cooks)
cooks[0].describe()



# So, we will take Model 2 because in this model RMSE is low and Rsquare is high.
# our model is above than 90% it's excellent model.

