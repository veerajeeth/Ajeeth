

import pandas as pd
df=pd.read_csv("Salary_Data.csv")
df

###spilt the variable

X=df[["YearsExperience"]]
X
Y=df["Salary"]
Y

## EDA (Scatter plot)

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0],Y,color='red')
plt.ylabel("Salary")
plt.xlabel("YearsExperience")
plt.show()

##Model fitting
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y2=LR.predict(X)
Y2

##Calculate RMSE,R square

from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
mse=mean_squared_error(Y,Y2)
RMSE=np.sqrt(mse)
print("Root mean square value:",RMSE)
print("R square value:",r2_score(Y,Y2))


#log transformation
X=df[["YearsExperience"]]
X
Y=np.log(df["Salary"])
Y

## EDA (Scatter plot)

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0],Y,color='red')
plt.ylabel("Salary")
plt.xlabel("YearsExperience")
plt.show()

##Model fitting
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y2=LR.predict(X)
Y2

##Calculate RMSE,R square

from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
mse=mean_squared_error(Y,Y2)
RMSE=np.sqrt(mse)
print("Root mean square value:",RMSE)
print("R square value:",r2_score(Y,Y2))

#sqrt transformations 
X=df[["YearsExperience"]]
X
Y=np.sqrt(df["Salary"])
Y

## EDA (Scatter plot)

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0],Y,color='red')
plt.ylabel("Salary")
plt.xlabel("YearsExperience")
plt.show()

##Model fitting
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y2=LR.predict(X)
Y2

##Calculate RMSE,R square

from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
mse=mean_squared_error(Y,Y2)
RMSE=np.sqrt(mse)
print("Root mean square value:",RMSE)
print("R square value:",r2_score(Y,Y2))

#log transformation has low mse value so selecting low mse is better 





















Simple linear (Salary data).py
Displaying Simple linear (Delivery time).py.