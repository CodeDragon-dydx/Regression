
# coding: utf-8

# In[1]:

from sklearn import datasets


# In[4]:

diabetes = datasets.load_diabetes()


# In[5]:

diabetes


# In[6]:

print(diabetes.DESCR)


# In[7]:

#Feature Names

print(diabetes.feature_names)


# In[8]:

#Create X and Y data matrices

X = diabetes.data
Y = diabetes.target


# In[9]:

X.shape, Y.shape


# In[10]:

#Load dataset + Create X and Y data matrices

X, Y = datasets.load_diabetes(return_X_y = True)


# In[11]:

X.shape, Y.shape


# In[12]:

#####Data Split#####

#Import Library

from sklearn.model_selection import train_test_split


# In[13]:

#Perform 80/20 data split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


# In[15]:

#Data Dimension

X_train.shape, Y_train.shape


# In[16]:


X_test.shape, Y_test.shape


# In[18]:

###Linear Regression Model###

#Import Library

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score


# In[19]:

#Build Linear Regression

#Defines the regression model

model = linear_model.LinearRegression()


# In[20]:

model.fit(X_train, Y_train)


# In[21]:

#Apply trained model to make prediction (on test set)

Y_pred = model.predict(X_test)


# In[22]:

###Prediction Results###

#Print model performance

print('Coefficients: ', model.coef_)

print('Intercept: ', model.intercept_)

print('Mean Squared Error(MSE): %.2f'
     % mean_squared_error(Y_test, Y_pred))

print('Coefficient of Determination(R^2): %.2f'
     % r2_score(Y_test, Y_pred))


# In[ ]:

Y = 43.42(AGE)-260.62(SEX)+565.49(BMI)+----+151.94


# In[23]:

###Scatter Plots###

#Make Scatter Plot

Y_test


# In[24]:

Y_pred


# In[34]:

import seaborn as sns
import matplotlib

get_ipython().run_line_magic('matplotlib', 'inline')


# In[35]:

sns.scatterplot(Y_test,Y_pred)


# In[36]:

sns.scatterplot(Y_test,Y_pred, marker = '+')


# In[37]:

sns.scatterplot(Y_test,Y_pred, alpha = 0.5)


# In[38]:

sns.scatterplot(Y_test,Y_pred, alpha = 1)


# In[ ]:



