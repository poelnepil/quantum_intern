#!/usr/bin/env python
# coding: utf-8

# # Import the libraries

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import scale
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm


# # Data analysis

# In[14]:


data_train = pd.read_csv('internship_train.csv')
data_train


# In[15]:


data_train.shape


# In[16]:


data_train.info()


# In[17]:


data_train.describe()


# In[18]:


corelations = data_train.corr()
corelations


# In[19]:


correlation_map = corelations
plt.figure(figsize = (15, 10))
sns.heatmap(correlation_map, cbar = True)
plt.show()


# In[20]:


data_train.shape


# In[21]:


for c in data_train.columns:
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(15,5))
    sns.histplot(data_train[c], ax=axes[0])
    axes[0].set_title(f'{c} distribution')
    sm.qqplot(data_train[c], ax=axes[1], line = '45')
    axes[1].set_title(f'{c} Q-Q plot')


# # Building and choosing a model

# In[22]:


y = data_train.target.values         
X = data_train.drop(['target'], axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# **LinearRegression**

# In[23]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[25]:


y_pred = lr.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Rmse: {lr_rmse}')


# **Random Forest**

# In[26]:


rf = RandomForestRegressor(random_state=0)
rf.fit(X_train, y_train)


# In[27]:


y_pred = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Rmse: {rf_rmse}')


# **Gradient Boosting Regressor**

# In[28]:


gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gbr.fit(X_train, y_train)


# In[31]:


y_pred = gbr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse_gbr = np.sqrt(mse)
print(f'Rmse: {rmse_gbr}')


# # Model evaluation

# In[33]:


sns.set_style('dark')
plt.figure(figsize = (15, 5))

plt.bar(x = ['Linear', 'Random Forest', 'Gradient Boosting Regressor'], height = [lr_rmse, rf_rmse, rmse_gbr])

plt.xlabel('Model', fontsize = 10)
plt.ylabel('Root-mean-square deviation', fontsize = 10)
plt.title('Comparision of models', fontsize = 15, weight = 'bold')

plt.show()


# *So, as we can see, Random Forest is the best model in terms of rmse, so there is no need to normalize the data for the final model. But to get an even better rmse, we will evaluate the importance of the features.*

# # Assessing the importance of features

# In[41]:


importances = rf.feature_importances_
df = pd.DataFrame({ "importances" : importances}, index=data_train.columns[:-1])
df.set_index('importances')
df.plot.bar(color = 'red')


# # Building the final model

# In[35]:


y = data_train.target.values 
X = data_train[['6', '7']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


# In[36]:


rf_fmodel = RandomForestRegressor(random_state=0)
rf_fmodel.fit(X_train, y_train)


# In[37]:


y_pred = rf_fmodel.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', rmse)


# In[38]:


hidden_test_data = pd.read_csv('internship_hidden_test.csv')
X_hidden_test = hidden_test_data[['6', '7']]
y_hidden_pred = rf_fmodel.predict(X_hidden_test)


# In[39]:


res = pd.DataFrame({'predicted target': y_hidden_pred})
res.to_csv('model predictions.csv')


# In[40]:


res

