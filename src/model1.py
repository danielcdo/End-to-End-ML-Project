#!/usr/bin/env python
# coding: utf-8

# # Statistical Learnning Model
# 
# **Goals:**
#  - Quickly check the data using samples of maching learning models
#  - Grasp some insights on the data
#  - Data clearning
# 
# Use the the target labels to fit the first model - base model.

# Import libraries

# In[87]:


import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import time

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.tree import plot_tree


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')


# ## Load raw data with labels

# In[4]:


df = pd.read_csv('./raw_data/raw_data_with_labels.csv', index_col=0)

# select only data with labels
df = df[df['y'].notnull()]
print('Num of labelled data: ', df.shape)


# In[5]:


df.head()
df.tail()


# # 1. Data Cleanup
# 
# Using a NEW dataframe with data that's ready and clean to fit the ML model.

# In[6]:


# create a clean dataframe with the same indice on the original dataframe - raw data
df_clean = pd.DataFrame(index=df.index)


# In[7]:


df_clean['date'] = pd.to_datetime(df['upload_date'], format='%Y%m%d')

# note: format='%Y %m %d' shows the time; format='%Y%m%d' brings only YYYY-MM-DD - easy!


# In[8]:


df_clean['date']

# dtype: datetime64[ns] used by numpy and pands


# In[9]:


# columns views: make sure all NAN will be convert to 0 and the assing an integer data type
df_clean['views'] = df['view_count'].fillna(0).astype(int)


# In[10]:


df_clean.dtypes


# ## 2.Features
# 
# Create an unique features dataframe.
# 
# **Reason**: Align the feaatures dataframe with the most cleaning data - raw data collected & cleaned. The cleaning process can skip rows or columns.
# 

# In[95]:


# features: it's similar to df_clean, just an extra step
features = pd.DataFrame(index=df_clean.index)

# labels/targets
y = df['y'].copy()


# In[96]:


print('Features shape: {}'.format(shape(features)))
print('Labels shape: {}'.format(shape(y)))


# ## Important: sklearn can't use *date* as a feature.
# 
# Let's manipulate and create a feature using the raw date - **Num_views_per_day**.
# 
# Sklearn needs a number.

# In[97]:


# time_since_pub: time since the video was published. Random data choose. Use the date I created this code: fix date point

# np.timedelta64(1, 'D'): time delta in numpy. Difference in days
# we have data on a granually day meaning a difference less than a ay makes sense.
features['time_since_pub'] = (pd.to_datetime("2021-05-09") - df_clean['date']) / np.timedelta64(1, 'D')

# used features
features['views'] = df_clean['views']
features['views_per_day'] = features['views'] / features['time_since_pub']

features = features.drop(['time_since_pub'], axis=1)   # time_since_pub only used for the calculation

# time_since_pub as a feature may impact the model once the numbers seem to increase a lot and the end of the time serie.
# The training&validations datasets may not have a normal distributed values, thus an umbalaced feature weights.
# and random samples are important to train and fit a ml model


# In[98]:


features.head()
features.tail()


# In[99]:


### not working
#df_clean['date'].value_counts().plot(figsize=(20.10))

features.describe()


# ## 3. Fitting a baseline model
# 
# Let's try to split the train&validation datasets 50/50.
# 
# How are the 2 features view and views_per_day impacted the ML model? 
# A simple model with only 2 features do impact the way the YouTube videos will be selected?

# In[100]:


# check all data on df_clean
# pd.set_option('display.max_rows', 527)
# df_clean

median_date = df_clean['date'].quantile(0.5, interpolation="midpoint")
median_date


# In[101]:


# splitting features dataset - trying a 50/50 using the median date
# balanced dataset is important
Xtrain, Xval = features[df_clean['date'] < '2021-03-12 '], features[df_clean['date'] >= '2021-03-12 ']
ytrain, yval = y[df_clean['date'] < '2021-03-12 '], y[df_clean['date'] >= '2021-03-12 ']

Xtrain.shape, Xval.shape, ytrain.shape, yval.shape


# ## Decision Tree Model
# 
# **Parameters**:
#  - 2 depth only - keep simple
#  - class_weight='balanced' ==> positive (1) samples, and zero are unbalanced and may impacte the ML model

# In[104]:


# check number of 1 samples under train dataset
print('Positive samples - videos select: {}'.format(ytrain.mean() * 263))
print(' % of ositive samples - videos select: {}'.format(ytrain.mean() * 100))

# definitely unbalaced


# In[105]:


clf_dt = DecisionTreeClassifier(random_state=0, max_depth=2, class_weight='balanced')    # defined object


# ## Fitting the model against the train dataset

# In[106]:


clf_dt.fit(Xtrain, ytrain)


# In[114]:


print('ML model already trainded/fitted and ready to be used!')


# ## Predicting if a video has been select
# 
# Probability = 1
# 
# predict_proba: returns a numpy array with prob of zero and prob of 1

# In[107]:


pred = clf_dt.predict_proba(Xval)[:, 1]   # only 1


# ## Metrics - validating the model
# 

# In[108]:


# area of precision for decision tree

print('Baseline model - decision tree with only simple parameters: 2 level deep')
average_precision_score(yval, pred)


# ## IMPORTANT: any future model in PRD should have a greater value

# In[109]:


# area under curve of roc curve metric
roc_auc_score(yval, pred)


# ## Decision Tree Plot 
# 

# In[112]:


fig, ax = pylab.subplots(1,1, figsize=(15,15))
plot_tree(clf_dt, ax=ax, feature_names=Xtrain.columns)


# ## Quick glance of the machine learning model
# ## Overall details for the Decision Tree statistical model
# 
# The parameter **class_weight='balanced'** worked punishing the positive samples that contain less examples. The root node shows the value with the same number meaning that the model tries to balance the samples.
# 
# Videos with less than 20 views per day - root node - and with less the 1 view per day will not be recommended - y label gets zero.
# The model seesm to make sense once not popular video should not be recommended.
# 
# On the other hande, left side on the tree, show videos with more views per day - likely popular videos - and with more than ~2400 views will be recommended.
# However, 'popular videos' with more than 20 views per day but with no more than ~2400 will not be recommented - get ZERO label.
# 
# Please note that it's a first simple example model (2 levels deep and only 2 features) to try to get a baseline metric.

# In[ ]:




