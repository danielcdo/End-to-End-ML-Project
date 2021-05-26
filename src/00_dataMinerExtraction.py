#!/usr/bin/env python
# coding: utf-8

# # Data Miner
# ## Get raw data from YouTube.com - scrapying

# **Import packages**
#  - **youtube-dl** is an open-source download manager for video and audio from YouTube and over 1000 other video hosting websites.It is released under the Unlicense software license.
# 
# **Source**: <a href="https://yt-dl.org/">youtube_dl</a>

# In[22]:


import numpy as np
import pandas as pd

import youtube_dl


# ## Add a query list
# List used to search videos. URL will be stored in a json file.
# 
# youtube_dl will get videos using the query list.

# In[2]:


# search list
queries = ["machine+learning", "data+science", "kaggle"]


# In[3]:


# put a defensive: to avoid videos not available --> {"ignoreerrors": True}
ydl = youtube_dl.YoutubeDL({"ignoreerrors": True})      # defined object


# Loop through the queries items, extract YouTube page using ydl.extract_info function. All parsed data is stored on json format under **entries**. 
# 
# Add query items under **entries**. 

# In[24]:


result = []        # store all the data parsed in a list

# get the pages
for query in queries:
    
    # use extract_info from youtube_dl package to parse the data
    # pass the number of videos to fetch --> ytsearchdate10 (meaning: search by date the first 10 videos)
    # r: result object variable
    
    # double defensive
    try: 
    # avoid error sample: ERROR: This live event will begin in 40 hours.    
        r = ydl.extract_info("ytsearchdate1000:{}".format(query), download=False)

        # add query under the json parsed by youtube_dl. Original parsed data doesn't contain the query parameters
        for entry in r['entries']:
            if entry is not None:                 # if the list contains an empty video
                entry['query'] = query            # add query items into the json data
        result += r['entries']                    # parsed data is stored under a list as 'entries' - json file!
    except:
        pass


# As a first attempt - raw data - duplicated videos are being parsed. 
# 
# Cleanup will be done down on data pipeline...not a big deal right now!
# 

# In[25]:


# put a defensive: just in case any Not available video is on the list
result = [e for e in result if e is not None]

print('Num of videas parsed: ', len(result))


# In[39]:


# convert the data parsed from you tube into a Pandas dataframe
# cool libray that provides the data so organized...great!
# youtube_dl avoid to manully scrap the youbube webpage
df = pd.DataFrame(result)            # result: long string with all data parsed

pd.set_option('display.max_columns', 70)

df.head(2)


# In[31]:


# feature engineering - a little bit!

# convert upload_date column to a date format
df['upload_date'] = pd.to_datetime(df['upload_date'])

# will done this one later!
# add new feaure - delta time: time since video publication
#df['time_since_pub'] = (pd.to_datetime("2021-05-31") - df['upload_date']) / np.timedelta64(1, 'D')


#pd.set_option("display.max_columns", 60)
pd.set_option('display.max_columns', 70)


df.head(2)


# ## Save dataframe using feather
# Feather is a fast, lightweight, and easy-to-use binary file format for storing data frames.
# 

# In[40]:


# gets only the desire features
#columns = ['title', 'upload_date', 'view_count', 'time_since_pub']
columns = ['title', 'upload_date', 'view_count']                      # features to train the model

# nan values may break the feather data format...can't convert
df[columns].to_feather('./raw_data/raw_data.feather')


# save also as a csv to be use on the labelling step
df[columns].to_csv('./raw_data/raw_data_without_labels.csv')


# In[30]:


print('Data miner - parsed raw data completed!')


# # Quick comparison  between feather format and csv

# In[35]:


# %time df[columns].to_feather('./raw_data/test_feather_format.feather')

# %time df[columns].to_csv('./raw_data/test_csv_format.feather')


# Wall time: 38 ms
# Wall time: 126 ms

# In[36]:


# df_= pd.read_feather('./raw_data/raw_data.feather')


# In[37]:


# just checking the raw data
# df_


# In[ ]:




