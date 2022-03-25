###### solution of Q1) Multilinear Regression by using L1 and L2 Regularization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm

# Loading the Datasets

data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Lasso and Ridge regression\Life_expectencey_LR.csv")
data.columns

# remove the column which is not useful for prediction

data.drop(['Country','Year'], axis=1, inplace = True)
data.info()
data.isnull().sum()

# EDA

data.describe()

data1 = pd.get_dummies(data, columns=['Status'], drop_first = True)
data1.columns

data1 = data1.rename(columns = { 'Status_Developing':'status'})
data1["status"].value_counts()

data1.fillna(data1.median(), inplace=True)
data1.isna().sum()

a = data1.corr()

# Pairplot
sns.pairplot(data1)

data1.columns
# model building
model = sm.ols('Life_expectancy ~ Adult_Mortality + infant_deaths + Alcohol + percentage_expenditure + Hepatitis_B + Measles + BMI + under_five_deaths + Polio + Total_expenditure + Diphtheria + HIV_AIDS + GDP + Population + thinness + thinness_yr + Income_composition + Schooling + status', data =data1).fit()
model.summary()

pred = model.predict(data1)

res = pred - data1.Life_expectancy
rmse = np.sqrt(np.mean(res*res))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###

from sklearn.linear_model import Lasso

lasso = Lasso(alpha =0, normalize = True)

lasso.fit(data1.iloc[:, 1:], data1.Life_expectancy)

# coefficient values for all independent variables

lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(data1.columns[1:]))
plt.xticks(rotation = 90)
lasso.alpha

pred_lasso = lasso.predict(data1.iloc[:, 1:])

# Adjusted R square
lasso.score(data1.iloc[:, 1:], data1.Life_expectancy)

# Rmse 
res_lasso = pred_lasso - data1.Life_expectancy

rmse_lasso = np.sqrt(np.mean(res_lasso* res_lasso))
rmse_lasso

### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 0 , normalize = True)

ridge.fit(data1.iloc[:,1:], data1.Life_expectancy)

ridge.coef_
ridge.intercept_

plt.bar(height = pd.Series(ridge.coef_), x= pd.Series(data1.columns[1:]))
plt.xticks(rotation = 90)

pred_ridge = ridge.predict(data1.iloc[:, 1:])
pred_ridge

# Adjusted R square
ridge.score(data1.iloc[:,1:], data1.Life_expectancy)

res_ridge = pred_ridge - data1.Life_expectancy
rmse_ridge = np.sqrt(np.mean(res_ridge*res_ridge))
rmse_ridge

###### ElasticNet Method #########
from sklearn.linear_model import ElasticNet
enet = ElasticNet( alpha = 0, normalize = True)

enet.fit(data1.iloc[:,1:], data1.Life_expectancy)

enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(data1.columns[1:]))
plt.xticks(rotation = 90)

pred_enet  = enet.predict(data1.iloc[:, 1:])

enet.score(data1.iloc[:,1:], data1.Life_expectancy)

res_enet = pred_enet - data1.Life_expectancy
rmse_enet = np.sqrt(np.mean(res_enet*res_enet))
rmse_enet 

####################
# Lasso Regression

from sklearn.model_selection import GridSearchCV
lasso = lasso()

parameter = { 'alpha':[1e-15,1e-10,1e-8, 1e-4, 1e-3, 1e-2, 1,5,10,20]}

lasso_reg = GridSearchCV(lasso, parameter, scoring = 'r2', cv = 5)
lasso_reg.fit(data1.iloc[:,1:], data1.Life_expectancy)

lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred_g = lasso_reg.predict(data1.iloc[:,1:])

# adjusted R square
lasso_reg.score(data1.iloc[:, 1:], data1.Life_expectancy)

res_lass_g =  lasso_pred_g - data1.Life_expectancy
rmse_lasso_g = np.sqrt(np.mean(res_lass_g*res_lass_g))
rmse_lasso_g

# Ridge Regression

ridge = Ridge()

parameter = { 'alpha':[1e-15,1e-10,1e-8, 1e-4, 1e-3, 1e-2, 1,5,10,20]}

ridge_reg = GridSearchCV(ridge, parameter, scoring = 'r2', cv = 5)
ridge_reg.fit(data1.iloc[:,1:], data1.Life_expectancy)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred_g = ridge_reg.predict(data1.iloc[:,1:])

# adjusted R square
ridge_reg.score(data1.iloc[:, 1:], data1.Life_expectancy)

res_ridge_g =  ridge_pred_g - data1.Life_expectancy
rmse_ridge_g = np.sqrt(np.mean(res_ridge_g*res_ridge_g))
rmse_ridge_g

# ElasticNet Regression

enet  = ElasticNet()
parameter = { 'alpha':[1e-15,1e-10,1e-8, 1e-4, 1e-3, 1e-2, 1,5,10,20]}

enet_reg = GridSearchCV( enet, parameter, scoring = 'r2', cv=5)
enet_reg.fit(data1.iloc[:,1:], data1.Life_expectancy)

enet_reg.best_params_
enet_reg.best_score_

enet_pred_g = enet_reg.predict(data1.iloc[:,1:])

enet_reg.score(data1.iloc[:,1:], data1.Life_expectancy)

res_enet_g = enet_pred_g - data1.Life_expectancy
rmse_enet_g = np.sqrt(np.mean(res_enet_g*res_enet_g))
rmse_enet_g
