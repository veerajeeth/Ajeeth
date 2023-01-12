
import numpy as np
import pandas as pd
df=pd.read_csv("ToyotaCorolla.csv",encoding="latin1")
df

# sort the data for your comfort
dfn = pd.concat([df.iloc[:,2:4],df.iloc[:,6:7],df.iloc[:,8:9],df.iloc[:,12:14],df.iloc[:,15:18]],axis=1)
dfn

# rename the data for your comfort
dfnw = dfn.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)
dfnw

dfnw[dfnw.duplicated()]

dfnew = dfnw.drop_duplicates().reset_index(drop=True)
dfnew
dfnew.describe()

# correlation 
dfnew.corr()

# step2: split the Variables in  X and Y's

# model 1
X = dfnew[["Age"]]

# Model 2
X = dfnew[["Age","Weight"]]

# Model 3
X = dfnew[["Age","Weight","KM"]]

# Model 4
X = dfnew[["Age","Weight","KM","HP"]]

# Model 5
X = dfnew[["Age","Weight","KM","HP","QT"]]

# Model 6
X = dfnew[["Age","Weight","KM","HP","QT","Doors"]]

# Model 7
X = dfnew[["Age","Weight","KM","HP","QT","Doors","CC"]]

# Model 8
X = dfnew[["Age","Weight","KM","HP","QT","Doors","CC","Gears"]]

# Target
Y = dfnew["Price"]

# scatter plot between each x and Y  
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(dfnew)
   
#==================================
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

# step 7 model deletion 
# Model Deletion Diagnostics
## Detecting Influencers/Outliers

## Cook’s Distance

lmreg_influence = lmreg.get_influence()
(cooks, pvalue) = lmreg_influence.cooks_distance

cooks = pd.DataFrame(cooks)
cooks[0].describe()