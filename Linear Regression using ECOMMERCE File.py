
# coding: utf-8

# ## Linear Regression- ECOMMERCE

# ### Project Description:
# 
# You got some contract with an Ecommerce company based in New York City that sells clothing online but also have in-store style and clothing advice sessions.
# 
# Customers come in to the store, have sessions/meetings with a personal stylist,that they can go home and order either on a mobile app or website for the clothes they want.
# 
# The company is trying to decide whether to focus their efforts on their mobile app experience or their website.
# They've hired you on contract to help them figure out!

# ### Imports

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data

# We'll work with the Ecommerce Customers csv file from the company. It has Customer info, such as Email,Address and their color Avatar. Then it also has numerical value columns.
# 
# - Avg.Session Length: Average session of in-store advice sessions.
# - Time on App: Average time spent on App in minutes.
# - Time on Website: Average time spent on Website in minutes.
# - Length of Membership:How many years the customer has been a member.
# 
# **Read in the Ecommerce Customers csv file as a DataFrame called customers.**

# In[4]:


customers = pd.read_csv('Ecommerce Customers')


# In[5]:


customers.head()


# In[6]:


customers.describe()


# In[7]:


customers.info()


# ## Exploratory Data analysis
# 
# #### Let's explore the data!
# 
# For the rest of the exercise we will only be using the numerical data of the csv file.
# 
# **Use Seaborn to create a joinplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?**

# In[10]:


sns.jointplot(data = customers, x ='Time on Website', y='Yearly Amount Spent')


#    **Do the same but with the Time on App column instead.**

# In[14]:


sns.jointplot(data=customers,x='Time on App', y='Length of Membership', kind='hex')


# In[16]:


sns.pairplot(customers)


# ### Based off this plot what looks to be the most correlated feature with yearly amount spent?
# 
# ### Answer: Length of Membership

# ---

# ### Create a linear model plot(using seaborn's Implot) of Yearly Amount Spent vs Length of Membership

# In[23]:


sns.lmplot(x ='Length of Membership', y ='Yearly Amount Spent', data=customers)


#  ---

# ## Training and Testing Data
# 
# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
# 
# **Set a Variable X equal to the numerical features of the customers and a variable Y equal to the "Yearly Amount Spent" column.**

# In[24]:


customers.columns


# In[25]:


Y = customers['Yearly Amount Spent']


# In[27]:


X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]


# **Use Cross_validation,train_test_split from sklearn to split the data into training and testing sets. Set test size=0.3 and random state=101**

# In[28]:


from sklearn.model_selection import train_test_split


# In[33]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)


# ---

# ## Training the Model
# 
# Now its time to train our model on our trianing data!
# 
# **Import LinearRegression from sklearn.linear_model**

# In[34]:


from sklearn.linear_model import LinearRegression


# In[35]:


regressor = LinearRegression()


# **Train/fit regressor on the training data.**

# In[36]:


regressor.fit(X_train, Y_train)


# **Print out the coefficients of the model**

# In[37]:


regressor.coef_


# ## Predicting Test Data
# 
# Now that we have fit our model, lets evaluate its performance by predicting off the test values!
# 
# **Use regressor.predict() to predict off the X_test of the data.**

# In[38]:


predictions = regressor.predict(X_test)


# **Create a Scatterplot of the real test values versus the predicted values**

# In[40]:


plt.scatter(Y_test,predictions)
plt.xlabel('Y_test(True Values)')
plt.ylabel('Predicted Values')


# ## Evaluating the model
# 
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score(R^2).
# 
# **Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.Refer to Wikipedia for the formulas.**

# In[41]:


from sklearn import metrics


# In[42]:


print('MAE', metrics.mean_absolute_error(Y_test, predictions))
print('MSE', metrics.mean_squared_error(Y_test, predictions))
print('RMSE', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))


# In[43]:


metrics.explained_variance_score(Y_test, predictions)


# ## Residuals
# 
# You should have gotten a very good model with a good fit. 
# Lets quickly explore the residuals to make sure everything was okay with our data.
# 
# **Plot a histogram of the residualas and make sure it looks normally distributed. Use either seaborn displot, or just plt.hist()**

# In[45]:


sns.distplot((Y_test-predictions), bins=50)


# ## Conclusion
# 
# We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important. Let's see if we can interpret the coefficients at all to get an idea.
# 
# **Recreate the dataframe below.**

# In[47]:


cdf = pd.DataFrame(regressor.coef_, X.columns, columns=['Coeff'])
cdf


# In[48]:


cdf.to_csv('Results.csv')


# #### How can you interpret these coefficients?
# 
# Basically we can interpret one at a time.
# 
# For instance, Avg.Session Length for every increase in 1 value there will be increase in 26$ approximately.
# 
# Similary for increase in Time on App there will be increase of 38$ spent per year.
# 
# Time on Website will be 0.19$ increase  spent per year.
# 
# Length of Membership for every increase there will 61.27$ increase per year.

# #### Do you think the company should focus more on their mobile app or on their website?
# 
# The coefficients showing that We could develop the website to catch up the mobile apps.Since the website needs much efforts than the mobile apps.
# 
# Or, we should focus more on mobile app since its already working better. 
