# **Status:** In Progress

# Project Overview: create a recommendation system using videos from YouTube.com
# Final product: recommentation video system hosted on a cloud service 

**Key steps to be noticed:**
 - add a threshold or videos ranking.
 - the final product will be a web app with videos links.
 - add and evaluate metrics to the problem:
   - 1st metric: from the top N videos, how many videos the user will be added to the list to be watched later?
   - 2nd metric: how long the video needs to be to be selected?

Possible issues:
 - feature noises: how long the user will take to select a video?
   This time may be correlated with the recommendation system.


Possible questions to be asked:
- choice the 'correct' problem'. Is it feasiable? Can it be done in a short period of time?
- minimum features required: videos attributes and a label. Are the features valuable?
- simple search query that contain - machine, learning, kaggle, and data science. Would be the sufficient to predict to recommend new videos with a high prediction score?

# Project in a nutshell

### 1. Problem well defined
  - Streamline the time to select a video to watch using key words on You Tube.
  - Automated process to avoid seaching for new videos.
  - Use Data Science techniques be used to solve this problem.
  - Deploty the final solution in production, cloud enviroment.

### 2. Data Miner and Exploratory Data Analysis

The data miner was done using the package ##youtube_dl##. Scrapying the You Tube page and parsing data features. Raw data selected and saved on feather data format.
The feather seems to be fast, lightweight, and easy-to-use binary file format for storing data frames.

 -  key queries to parse the videos data: machine+learning, data+science, and kaggle.
 
 Note that 1000 videos were selected.
 
 ```
 # use extract_info from youtube_dl package to parse the data
# pass the number of videos to fetch --> ytsearchdate1000 (meaning: search by date the first 1000 videos)
# r: result object variable
 r = ydl.extract_info("ytsearchdate1000:{}".format(query), download=False)

```
After **raw data** miner step, then the **labelling** was used to classify videos as: #yes - watch#, and #not - do not watch#. Manual process, but also an active learning method was used. The active learning was used to select videos in which the the machine learning model was not able to predict with a high probability, around 50%.
The aim was to try to 'push' the hyperplan for samples on the edge.

### 3. Exploratory Data Analysis - EDA




