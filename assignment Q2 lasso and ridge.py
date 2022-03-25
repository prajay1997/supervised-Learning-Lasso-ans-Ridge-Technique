###### solution of Q2) Multilinear Regression by using L1 and L2 Regularization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm

# Loading the Datasets

data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Lasso and Ridge regression\ToyotaCorolla (1).csv")
data.columns

# remove the column which is not useful for prediction

data.drop(['Id','Model','Mfg_Month','Mfg_Year','Fuel_Type','Met_Color', 'Color', 'Automatic', 'Cylinders','Mfr_Guarantee',
       'BOVAG_Guarantee', 'Guarantee_Period', 'ABS', 'Airbag_1', 'Airbag_2',
       'Airco', 'Automatic_airco', 'Boardcomputer', 'CD_Player',
       'Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio',
       'Mistlamps', 'Sport_Model', 'Backseat_Divider', 'Metallic_Rim',
       'Radio_cassette', 'Tow_Bar'], axis=1, inplace = True)


data.columns

data.info()
data.isnull().sum()

# EDA

data.describe()

data1 = data.rename(columns = {'Price':'price', 'Age_08_04':'age','Quarterly_Tax':'quarterly_tax'} )

data1.corr()

# Pairplot
sns.pairplot(data1)
data1.columns
# model building
model = sm.ols('price ~ age + KM + HP + cc + Doors + Gears + quarterly_tax + Weight', data =data1).fit()
model.summary()

pred = model.predict(data1)

res = pred - data1.price
rmse = np.sqrt(np.mean(res*res))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###

from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.15 , normalize = True)

lasso.fit(data1.iloc[:, 1:], data1.price)

# coefficient values for all independent variables

lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(data1.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(data1.iloc[:, 1:])

# Adjusted R square
lasso.score(data1.iloc[:, 1:], data1.price)

# Rmse 
res_lasso = pred_lasso - data1.price

rmse_lasso = np.sqrt(np.mean(res_lasso* res_lasso))
rmse_lasso

### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 0.15 , normalize = True)

ridge.fit(data1.iloc[:,1:], data1.price)

ridge.coef_
ridge.intercept_

plt.bar(height = pd.Series(ridge.coef_), x= pd.Series(data1.columns[1:]))

pred_ridge = ridge.predict(data1.iloc[:, 1:])
pred_ridge

# Adjusted R square
ridge.score(data1.iloc[:,1:], data1.price)

res_ridge = pred_ridge - data1.price
rmse_ridge = np.sqrt(np.mean(res_ridge*res_ridge))
rmse_ridge

###### ElasticNet Method #########
from sklearn.linear_model import ElasticNet
enet = ElasticNet( alpha = 0, normalize = True)

enet.fit(data1.iloc[:,1:], data1.price)

enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(data1.columns[1:]))

pred_enet  = enet.predict(data1.iloc[:, 1:])

enet.score(data1.iloc[:,1:], data1.price)

res_enet = pred_enet - data1.price
rmse_enet = np.sqrt(np.mean(res_enet*res_enet))
rmse_enet 

####################
# Lasso Regression

from sklearn.model_selection import GridSearchCV
lasso = lasso()

parameter = { 'alpha':[1e-15,1e-10,1e-8, 1e-4, 1e-3, 1e-2, 1,5,10,20]}

lasso_reg = GridSearchCV(lasso, parameter, scoring = 'r2', cv = 5)
lasso_reg.fit(data1.iloc[:,1:], data1.price)

lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred_g = lasso_reg.predict(data1.iloc[:,1:])

# adjusted R square
lasso_reg.score(data1.iloc[:, 1:], data1.price)

res_lass_g =  lasso_pred_g - data1.price
rmse_lasso_g = np.sqrt(np.mean(res_lass_g*res_lass_g))
rmse_lasso_g

# Ridge Regression

ridge = Ridge()

parameter = { 'alpha':[1e-15,1e-10,1e-8, 1e-4, 1e-3, 1e-2, 1,5,10,20]}

ridge_reg = GridSearchCV(ridge, parameter, scoring = 'r2', cv = 5)
ridge_reg.fit(data1.iloc[:,1:], data1.price)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred_g = ridge_reg.predict(data1.iloc[:,1:])

# adjusted R square
ridge_reg.score(data1.iloc[:, 1:], data1.price)

res_ridge_g =  ridge_pred_g - data1.price
rmse_ridge_g = np.sqrt(np.mean(res_ridge_g*res_ridge_g))
rmse_ridge_g

# ElasticNet Regression

enet  = ElasticNet()
parameter = { 'alpha':[1e-15,1e-10,1e-8, 1e-4, 1e-3, 1e-2, 1,5,10,20]}

enet_reg = GridSearchCV( enet, parameter, scoring = 'r2', cv=5)
enet_reg.fit(data1.iloc[:,1:], data1.price)

enet_reg.best_params_
enet_reg.best_score_

enet_pred_g = enet_reg.predict(data1.iloc[:,1:])

enet_reg.score(data1.iloc[:,1:], data1.price)

res_enet_g = enet_pred_g - data1.price
rmse_enet_g = np.sqrt(np.mean(res_enet_g*res_enet_g))
rmse_enet_g
