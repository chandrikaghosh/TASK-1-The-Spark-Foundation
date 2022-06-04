#!/usr/bin/env python
# coding: utf-8

# # Prediction using Supervised ML

# ##### Predicting the percentage of students based on the numbers of hours of study

# ##Data- http://bit.ly/w-data

# In[1]:


#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#reading the data set
url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print(df)


# In[3]:


#exploring the data
print(df.shape)

df.info()


# In[4]:


df.describe()


# Let us draw a scatter diagram to see the relationship between percentage the students have achieved and hours they study.

# In[5]:


#plotting the distribution of scores
df.plot(kind = "scatter", x = "Hours", y = "Scores")
plt.show()


# From the above plot we can see that the percentage of students are positively related to the number of hours they study. So we can say that more the number of hrs students study better their percentage gets.We can also infer that the variables have a fairly linear relation

# In[6]:


#to check that whether the variables are positively related we check the correlation coefficient.
df.corr(method = "spearman")


# The correlation coefficient is close to 1 i.e. 0.97, so we can say that the variables are positively related.

# #### Starting with Linear Regression

# In[7]:


#preparing the data
x = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values


# In[8]:


print(x)
print(y)


# In[9]:


from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                            test_size=0.2, random_state=0) 


# In[10]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)
print("training complete.")


# In[11]:


#plotting the regression line 
line = reg.coef_*x + reg.intercept_


# In[12]:


#plotting the test data 
plt.scatter(x, y)
plt.plot(x, line)
plt.title("Actual vs Predicted")
plt.ylabel("Scores")
plt.xlabel("Hours")
plt.show()


# In[13]:


print(x_test) 


# In[14]:


y_pred = reg.predict(x_test)


# In[15]:


#comparing actual and predicted scores
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[16]:


#calculating accuracy of the model.
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 


# Small value of Mean Absolute Error states that there is very less chance of error in forecasting.

# In[17]:


#considering a numerical value for hours to see the predicted value of score.
Hours = [9.25]
answer = reg.predict([Hours])
print("Score = {}".format(answer))


# When a student study for 9.25 hours then the predicted score is 93.69.

# In[ ]:




