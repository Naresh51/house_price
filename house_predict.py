#!/usr/bin/env python
# coding: utf-8

# # House Price 

# ![download.jpg](attachment:download.jpg)

# ### Goal of the project <br>
# predict the price of a house by its features.if you are buyer or seller of the house but you do not know the exact price of the house , so machine learning algorithms can help to predict the price of the house just providing features of the target house.

# #### About Dataset
# This dataset provides comprehensive information for house price prediction, with 13 column names:
# 
# Price: The price of the house.<br>
# Area: The total area of the house in square feet.<br>
# Bedrooms: The number of bedrooms in the house.<br>
# Bathrooms: The number of bathrooms in the house.<br>
# Stories: The number of stories in the house.<br>
# Mainroad: Whether the house is connected to the main road (Yes/No).<br>
# Guestroom: Whether the house has a guest room (Yes/No).<br>
# Basement: Whether the house has a basement (Yes/No).<br>
# Hot water heating: Whether the house has a hot water heating system (Yes/No).<br>
# Airconditioning: Whether the house has an air conditioning system (Yes/No).<br>
# Parking: The number of parking spaces available within the house.<br>
# Prefarea: Whether the house is located in a preferred area (Yes/No).<br>
# Furnishing status: The furnishing status of the house (Fully Furnished, Semi-Furnished, Unfurnished).<br>
# 

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv(r'C:\Users\hp\Desktop\Housing.csv')


# In[4]:


df


# In[5]:


df.shape


# In[6]:


df.sample(5)


# In[7]:


df.info()


# The column 'price' is numeric.<br>
# The column 'area(m2)' is numeric.<br>
# The column 'bedrooms' is numeric.<br>
# The column 'bathrooms' is numeric.<br>
# The column 'stories' is numeric.<br>
# The column 'mainroad' is catagorical <br>.
# The column 'guestroom' is catagorical.<br>
# The column 'basement' is catagorical.<br>
# The column 'hotwaterheating' is catagorical.<br>
# The column 'airconditioning' is catagorical.<br>
# The column 'parking' is numeric.<br>
# The column 'prefarea' is catagorical.<br>
# The column 'furnishingstatus' is catagorical.<br>

# In[8]:


df.describe()


# In[9]:


df.isnull().sum()


# thers is no missing in this data set

# In[10]:


df.duplicated().sum()


# there is no duplicate in this data set

# In[11]:


df.corr()


# #### correlation <br>
# price  most strongly cor realtion with area <br>
# area most strongly cor realtion with price<br>
# bedrooms most strongly cor relation with stories<br>
# bathrooms most strongly  cor realtion with price<br>
# stories most strongly cor realtion with parking<br>
# parkings most strongs cor realtion with price

# ### EDA

# In[12]:



import matplotlib.pyplot as plt


# In[13]:


import seaborn as sns


# In[14]:


import warnings
warnings.filterwarnings('ignore')


# In[222]:


sns.pairplot(df)
plt.show()


# #### correlation 

# In[266]:


sns.heatmap(df.corr(),annot=True)


# price  most strongly cor realtion with area <br>
# area most strongly cor realtion with price<br>
# bedrooms most strongly cor relation with stories<br>
# bathrooms most strongly  cor realtion with price<br>
# stories most strongly cor realtion with parking<br>
# parkings most strongs cor realtion with price

# #### finding outlier in numerical columns in the data set ?

# In[267]:


# Outlier Analysis
fig, axs = plt.subplots(2,3, figsize = (10,5))
plt1 = sns.boxplot(df['price'], ax = axs[0,0])
plt2 = sns.boxplot(df['area'], ax = axs[0,1])
plt3 = sns.boxplot(df['bedrooms'], ax = axs[0,2])
plt1 = sns.boxplot(df['bathrooms'], ax = axs[1,0])
plt2 = sns.boxplot(df['stories'], ax = axs[1,1])
plt3 = sns.boxplot(df['parking'], ax = axs[1,2])

plt.tight_layout()


# #### finding outlier of catagorical columns with respect to house price column?

# In[268]:


plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'mainroad', y = 'price', data = df)
plt.subplot(2,3,2)
sns.boxplot(x = 'guestroom', y = 'price', data = df)
plt.subplot(2,3,3)
sns.boxplot(x = 'basement', y = 'price', data = df)
plt.subplot(2,3,4)
sns.boxplot(x = 'hotwaterheating', y = 'price', data = df)
plt.subplot(2,3,5)
sns.boxplot(x = 'airconditioning', y = 'price', data = df)
plt.subplot(2,3,6)
sns.boxplot(x = 'furnishingstatus', y = 'price', data = df)
plt.show()


# #### how does the area effect the price?

# In[223]:


plt.figure(figsize=(5,3))
sns.scatterplot(x=df.area,y=df.price)
plt.title("Area VS Price")
plt.show()


# #### **Price comparision with bedroom,bathroom,stories and parking
# 
# 

# In[224]:


fig=plt.subplots(2,2,figsize=(20,10))

plt.subplot(2,2,1)

sns.boxplot(x="bedrooms", y="price", data=df)
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.title("Bedrooms vs. Price")

plt.subplot(2,2,2)
sns.boxplot(x="bathrooms", y="price", data=df)
plt.xlabel("bathrooms")
plt.ylabel("Price")
plt.title("Bedrooms vs. Price")

plt.subplot(2,2,3)

sns.boxplot(x="stories", y="price", data=df)
plt.xlabel("stories")
plt.ylabel("Price")
plt.title("Bedrooms vs. Price")

plt.subplot(2,2,4)

sns.boxplot(x="parking", y="price", data=df)
plt.xlabel("parking")
plt.ylabel("Price")
plt.title("parking vs. Price")

plt.show()


# #### **furnishingstatus with the price
# 
# 

# In[15]:


import plotly.express as px

fig=px.scatter(x=df['furnishingstatus'],y=df['price'],color=df['furnishingstatus'])
fig.show()


# #### How does the number of bedrooms affect the sale price?

# In[226]:


# price convert into lakhs 
house_price=df['price']/1000000


# In[227]:


plt.figure(figsize=(5,3))
plt.bar(df.bedrooms,house_price)
plt.xlabel("No. of bedrooms")
plt.ylabel("House price in Lakhs")
plt.title("house price according to No. of bedrooms")
plt.show()


# #### How does the number of bathrooms affect the sale price?

# In[229]:


plt.figure(figsize=(5,3))
plt.bar(df.bathrooms,house_price)
plt.xlabel("No. of bathroomd")
plt.ylabel("House price in Lakhs")
plt.title("house price according to No. of bathrooms")
plt.show()


# #### How does the number of bathrooms and bedrooms affect the sale price?

# In[230]:


plt.figure(figsize=(5,4))
sns.scatterplot(x=df.area, y=house_price, hue=df.bathrooms, size=df.bedrooms)
plt.show()


# ## Data prepration Model

# In[16]:


df.columns


# In[17]:


df.head()


# independance variable='area', 'bedrooms', 'bathrooms', 'stories', 'mainroad','guestroom', 'basement', 'hotwaterheating', 'airconditioning','parking', 'prefarea','furnishingstatus'<Br>
# depandance variable= price

# ## Encoding

# In[18]:


# mainroad,guestroom,basement,hotwaterheating, airconditioning,prefarea, furnishingstatus are is ordinal categorial variable


# In[19]:


from sklearn.preprocessing import OrdinalEncoder


# In[20]:


oe=OrdinalEncoder(categories=[['no','yes']])
col=['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
for col_n in col:
  df[col_n]=oe.fit_transform(df[[col_n]])


# In[21]:


df.head()


# In[22]:


oren=OrdinalEncoder(categories=[['unfurnished','semi-furnished','furnished']])
df['furnishingstatus']=oren.fit_transform(df[['furnishingstatus']])


# In[23]:


df.head()


# In[24]:


df.sample(10)


# In[25]:


x=df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad','guestroom', 'basement', 'hotwaterheating', 'airconditioning','parking', 'prefarea','furnishingstatus']]
y=df['price']/100000
x['area']=x['area']/1000


# In[26]:


x


# In[27]:


y


# ## Spiliting Data into training and test set
# 

# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[30]:


print(x_train)
print(x_test)
print(y_train)
print(y_test)


# In[31]:


print("X_train:", x_train.shape)
print("X_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# In[32]:


sns.distplot(df)


# ## feature scaling

# In[33]:


from sklearn.preprocessing import MinMaxScaler


# In[34]:


mnc=MinMaxScaler()


# In[35]:


mnc.fit_transform(x_train)


# In[36]:


x_train_mnc=mnc.transform(x_train)
x_test_mnc=mnc.transform(x_test)


# In[37]:


x_train_df=pd.DataFrame(x_train_mnc,columns=x_train.columns)


# In[38]:


x_test_df=pd.DataFrame(x_test_mnc,columns=x_test.columns)


# In[39]:


x_train_df.describe().round(2)


# In[40]:



# Before scaling
sns.pairplot(x_train[['area']])
plt.title("Before Scaling")
plt.show()

# After scaling
sns.pairplot(x_train_df[['area']])
plt.title("After Scaling")
plt.show()


# ** thier is no change after scaling

# ## Applying Machine learning Model

# In[41]:


from sklearn.linear_model import LinearRegression 


# In[49]:



lr_scared=LinearRegression()


# In[50]:



lr_scared.fit(x_train_df,y_train)


# ## predict the value of home and test

# In[51]:


# Make predictions

predictions=lr_scared.predict(x_test_df)


# In[52]:


predictions


# In[53]:


y_test


# In[55]:


lr_scared.score(x_test_df,y_test)


# In[56]:


from sklearn.metrics import *


# In[57]:


MAE=mean_absolute_error(y_test,predictions)


# In[58]:


MSE=mean_squared_error(y_test,predictions)


# In[59]:


RMSE=np.sqrt(MSE)


# In[60]:


MAE,MSE,RMSE


# 
# 
# 
# 
# ## Implement Ridge and Lasso

# In[61]:


from sklearn.linear_model import Ridge ,Lasso


# In[63]:


rd=Ridge()
rd.fit(x_train_df,y_train)
rd.score(x_test_df,y_test)


# In[64]:


la=Lasso()
la.fit(x_train,y_train)
la.score(x_test_df,y_test),


# In[65]:


print("linear model predictions : ",lr_scared.score(x_test_df,y_test))
print('Ridge model prediction : ',rd.score(x_test_df,y_test))
print("Lasso model predictions : ",la.score(x_test_df,y_test))


# ### Save best model 

# In[66]:


import pickle


# In[69]:


if lr_scared.score(x_test_df,y_test)>rd.score(x_test_df,y_test) and lr_scared.score(x_test_df,y_test)>la.score(x_test_df,y_test):
    print("pick linear beacuse linear")
elif rd.score(x_test_df,y_test)>lr_scared.score(x_test_df,y_test) and rd.score(x_test_df,y_test)>la.score(x_test_df,y_test):
    print(" pick Ridge its best")
else:
    print("pick lasso")


# In[75]:


import pickle

with open("house", 'wb') as file:
    pickle.dump(rd, file)


# ### read file

# In[78]:


with open("house",'rb')as file:
    model=pickle.load(file)


# In[79]:


model


# In[81]:


pip install streamlit


# In[82]:


import streamlit as st


# In[83]:


st.title("hello streamlit")


# In[ ]:




