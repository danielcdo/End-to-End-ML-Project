# **Status:** In Progress

# Project Overview: create a recommendation system using videos from YouTube.com
# Final product: recommentation video system hosted on a cloud service 

**Key steps to be noticed:**
 - add a threshold or videos ranking.
 - the final product will be a web application.
 - add and evaluate metrics to the problem:
   - 1st metric: from the top N videos. How many videos will be added to the list?
   - 2nd metric: video size to be to be selected?
   - 3rd: does the number of views impact the model?
   - 4th: does the video title affect the model?

Possible issues:
 - feature noises: how long the user will take to select a video?
   This time may be correlated with the recommendation system.


Possible questions to be asked:
- choose the 'correct' problem'. Is it feasible? Can it be done in a short time?
- minimum features required: videos attributes and label. Are the features valuable?
- simple search query that contains - machine, learning, kaggle, and data science as keywords. Would be the features sufficient to get a high score recommending new videos?
- Will the feature engineering be requested by the statistical model?

# Project in a nutshell

### 1. Define the problem
  - Streamline the time to select videos using keywords on You Tube.
  - Process automation to avoid searching new videos.
  - Use Data Science techniques to solve this problem.
  - Deploy the final production solution in a cloud platform service.

### 2. Data Miner and Exploratory Data Analysis

The data miner was done using the package **youtube_dl**. Scrapying the You Tube page and parsing data attributes. Raw data selected and saved using a feather data format.
The feather format seems to be fast, lightweight, and easy-to-use binary file format for storing the data frames.

 -  key queries to parse the videos data: machine+learning, data+science, and kaggle.
 
 Note that 1000 videos were selected.
 
 ```
 # use extract_info from youtube_dl package to parse the data
# pass the number of videos to fetch --> ytsearchdate1000 (meaning: search by date the first 1000 videos)
# r: result object variable
 r = ydl.extract_info("ytsearchdate1000:{}".format(query), download=False)

```
After the **raw data** miner step, then the **labelling** was used to classify videos as: *yes - to watch*, and *not - do not watch*. It was a manual process, but also an active learning method was used.

The active learning was applied to select videos in which the the machine learning model was not able to predict with a high probability - around 50%. Mathemattically speaking, it's not a probability, but score to select a video.
The aim was to try to 'push' the hyperplan for samples on the edge. The machine learning model was having a *hard* time to predict *yes* or *no* and the active learning is a good approach for a situation where labelling may be too expensive.

On the feature engineering phase on the fly: 
- views_per_day
- vectorize title using TfidfVectorizer function to return a vectorized sparse matrix. It's an optimize matrix in Scipy where only values NOT equal to zero are returned.

### 3. Machine Learning Modelling

Decison Tree and Random Forest models used as a baseline models. The LighGBM was the first statistical learning model in production environment.

### 4.Final step - Deploy

Final deployment was done using Flask in saved in the Heroku cloud platform service.

