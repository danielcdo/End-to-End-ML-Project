#!/usr/bin/env python
# coding: utf-8

# # Statistical Learnning Model 
# 
# **Goals:**
#  - Apply active learning
#  - Quickly check the data using samples of maching learning models
#  - Grasp some insights on the data
#  - Data clearning
#  - Add more the title feature
# 
# Use the the target labels to fit the first model - base model.

# Import libraries

# In[1]:


#!conda update scikit-learn


# In[1]:


import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction import TfidfVectorizer
from sklearn.metrics import roc_auc_score, average_precision_score


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')


# ## Load raw data with labels

# In[ ]:


df = pd.read_csv('./raw_data/raw_data_with_labels.csv', index_col=0)

# select only data with labels
df = df[df['y'].notnull()]
print('Num of labelled data: ', df.shape)


# In[ ]:


df.head()
df.tail()


# # 1. Data Cleanup
# 
# Using a NEW dataframe with data that's ready and clean to fit the ML model.

# In[ ]:


# create a clean dataframe with the same indice on the original dataframe - raw data
df_clean = pd.DataFrame(index=df.index)


# In[ ]:


df_clean['date'] = pd.to_datetime(df['upload_date'], format='%Y%m%d')

# note: format='%Y %m %d' shows the time; format='%Y%m%d' brings only YYYY-MM-DD - easy!


# In[ ]:


df_clean['date']

# dtype: datetime64[ns] used by numpy and pands


# In[ ]:


# columns views: make sure all NAN will be convert to 0 and the assing an integer data type
df_clean['views'] = df['view_count'].fillna(0).astype(int)


# In[ ]:


df_clean.dtypes


# ## 2.Features
# 
# Create an unique features dataframe.
# 
# **Reason**: Align the feaatures dataframe with the most cleaning data - raw data collected & cleaned. The cleaning process can skip rows or columns.
# 

# In[ ]:


# features: it's similar to df_clean, just an extra step
features = pd.DataFrame(index=df_clean.index)

# labels/targets
y = df['y'].copy()


# In[ ]:


print('Features shape: {}'.format(shape(features)))
print('Labels shape: {}'.format(shape(y)))


# ## Important: sklearn can't use *date* as a feature.
# 
# Let's manipulate and create a feature using the raw date - **Num_views_per_day**.
# 
# Sklearn needs a number.

# In[ ]:


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


# In[ ]:


features.head()
features.tail()


# In[ ]:


### not working
#df_clean['date'].value_counts().plot(figsize=(20.10))

features.describe()


# ## 3. Fitting a baseline model
# 
# Let's try to split the train&validation datasets 50/50.
# 
# How are the 2 features view and views_per_day impacted the ML model? 
# A simple model with only 2 features do impact the way the YouTube videos will be selected?

# In[ ]:


# check all data on df_clean
# pd.set_option('display.max_rows', 527)
# df_clean

median_date = df_clean['date'].quantile(0.5, interpolation="midpoint")
median_date


# In[ ]:


# splitting features dataset - trying a 50/50 using the median date
# balanced dataset is important
# Xtrain, Xval = features[df_clean['date'] < '2021-03-12'], features[df_clean['date'] >= '2021-03-12']
# ytrain, yval = y[df_clean['date'] < '2021-03-12 '], y[df_clean['date'] >= '2021-03-12 ']

# need approach - mask parameter to select the data
mask_train = df_clean['date'] < '2021-03-12'
mask_val = df_clean['date'] >= '2021-03-12'

Xtrain, Xval = features[mask_train], features[mask_val]
ytrain, yval = y[mask_train], y[mask_val]

Xtrain.shape, Xval.shape, ytrain.shape, yval.shape


# ## Add the title feature
# 
# **Important**: 

# In[ ]:


title_vec = TfindfVectorized(min_df=2)


# ## Decision Tree Model
# 
# **Parameters**:
#  - 2 depth only - keep simple
#  - class_weight='balanced' ==> positive (1) samples, and zero are unbalanced and may impacte the ML model

# In[ ]:


# check number of 1 samples under train dataset
print('Positive samples - videos select: {}'.format(ytrain.mean() * 263))
print(' % of ositive samples - videos select: {}'.format(ytrain.mean() * 100))

# definitely unbalaced


# In[ ]:


clf_dt = DecisionTreeClassifier(random_state=0, max_depth=2, class_weight='balanced')    # defined object


# ## Fitting the model against the train dataset

# In[ ]:


clf_dt.fit(Xtrain, ytrain)


# In[ ]:


print('ML model already trainded/fitted and ready to be used!')


# ## Predicting if a video has been select
# 
# Probability = 1
# 
# predict_proba: returns a numpy array with prob of zero and prob of 1

# In[ ]:


pred = clf_dt.predict_proba(Xval)[:, 1]   # only 1


# ## Metrics - validating the model
# 

# In[ ]:


# area of precision for decision tree

print('Baseline model - decision tree with only simple parameters: 2 level deep')
average_precision_score(yval, pred)


# ## IMPORTANT: any future model in PRD should have a greater value

# In[ ]:


# area under curve of roc curve metric
roc_auc_score(yval, pred)


# ## Decision Tree Plot 
# 

# In[ ]:


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




