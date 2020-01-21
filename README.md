Spotify Genre Prediction
===

# Index
1. Project definition
2. Related projects
3. Data acquisition
4. Study of the dataset
5. Study of the features
6. Creating new dataset
7. Assumptions
8. Analysis
9. Prediction
10. Evaluation
11. Conclusion

# Project definition

Spotify provides an API which returns JSON metadata about music artists, albums, and tracks, directly from their DataBase
One of their endpoints enable the user to get Audio Features for a Track. This is, to obtain the high level characteristics that define a song.

In this project we will work on predicting the genre of a song based on those audio analysis features. The main purpose ot this work is to analyse if the features extracted by Spotify (by unknown audio analysis criteria) allow us to group music genres only taking that information into consideration.

First, we will have to find the list of tracks that we will use as training/testing data.
Then, we will get the audio features of each track and create a new dataset.
Lastly, we will perform the analysis and prediction of the genres and evaluate accuracy by applying different models and techniques.

Since this is a project that I created to learn more about Data Science, I will add/modify solutions according to what I learn over time.

Firstly, we will start by pointing out some assumptions without knowing anything about the data or the problem:
- Spotify generates a set of audio features based on audio analysis that they perform. Some audio features will have more influence than others in the accuracy of the genre prediction, and we will need to perform data selection and understand what impact has each data feature
- There are a wide number of genres that a song can be categorised as. We will generalise music genres into a few of them in order to simplify the problem
- Some genres will have similar audio features

# Research related projects

We collect a list of similar projects that serve as inspiration: 

Predicting hit songs: 
https://pdfs.semanticscholar.org/e6cc/edb50d2c2b01bca108cb090943e86fb58135.pdf
https://towardsdatascience.com/song-popularity-predictor-1ef69735e380

Using lyrics to predict the genre of a track:
https://towardsdatascience.com/how-we-used-nltk-and-nlp-to-predict-a-songs-genre-from-its-lyrics-54e338ded537
http://cs229.stanford.edu/proj2017/final-reports/5242682.pdf

We haven't found similar project to the one that will be developed (Genre Classification of Spotify Songs using Lyrics,
Audio Previews, and Album Artwork is the closest to this).

# Data acquisition

The objective of this project is to predict the genre of a track based on its features.
Features will be obtained from the Spotify Restful API.

A RESTful API is an application program interface (API) that uses HTTP requests to GET, PUT, POST and DELETE data [[1]](https://searchmicroservices.techtarget.com/definition/RESTful-API ).

We will use the following endpoint to get the features: 
https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/

The study of the Spotify features is done in the section "Study of features".

Next step will be to get the tracks dataset. For this, we will define a number of music genres and we will get a set of X tracks for each music genre. 

First issue that we encounter here is that we can't get the genre of a track with the Spotify API, but we can get the genre of an artist. However, not all the songs of an artist that plays "pop" are "pop". We could generalise but, would it be okay to do that?

If we use the Spotify Restful API to check out the genre of an artist like "Arctic Monkeys" we find the following:

    "genres": [
        "garage rock",
        "indie rock",
        "modern rock",
        "permanent wave",
        "rock",
        "sheffield indie"
      ]

Spotify tags artists with several "subgenres" which makes it difficult to reduce an artist to a simple genre.

Therefore we have two options to get our data:
1. Find a dataset with the tags defined
2. Define a number of genres that we want to study. Take a list of songs. For each song, find genre in spotify. If it matches one of the genres selected, store that song in the dataset

In this project we will select the first option, leaving the second one for future work.
We will use the following dataset: https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics

In this dataset we have the following information:

![](https://i.imgur.com/DVVZVmu.png)

For our project, it will be useful to consider:
- Genre
- Track
- Artist

For each track we will get the features using the Spotify API and we will combine both into a final dataset that will be used for the predictions.

First we will study the initial dataset.

# Study of the dataset

Doing a first count of the dataset genres, we find the following:

    [('Rock', 131377),
     ('Pop', 49444),
     ('Hip-Hop', 33965),
     ('Not Available', 29814),
     ('Metal', 28408),
     ('Other', 23683),
     ('Country', 17286),
     ('Jazz', 17147),
     ('Electronic', 16205),
     ('R&B', 5935),
     ('Indie', 5732),
     ('Folk', 3241)]

We discard "Not Available" and "Other" as they are not meaningful for the prediction.
We will also drop "Indie" and "Folk" to simplify the problem.

Therefore, we end up with 8 genres: 

    list_genres = ["Rock", "Pop", "Hip-Hop", "Metal", "Country", "Jazz", "Electronic", "R&B"]

This provides enough categorical variety.

First, we will perform data cleaning: drop rows with any NaN value:

    [('Rock', 109235),
     ('Pop', 40466),
     ('Hip-Hop', 24850),
     ('Not Available', 23941),
     ('Metal', 23759),
     ('Country', 14387),
     ('Jazz', 7970),
     ('Electronic', 7966),
     ('Other', 5189),
     ('R&B', 3401),
     ('Indie', 3149),
     ('Folk', 2243)]

Rock is the most common data category. In order to avoid data imbalance, we will take a subset of 1000 random songs for each genre.
By randomising we avoid having the same artist repeated a lot, assuming that more data variety may improve the prediction
We could take 1000 songs (since the less frequent is Folk that has 2243) but we are curious to check what will be the accuracy in prediction with a lower volume of data.
//TODO taking 2000 songs
//TODO comment client id and secret key
//TODO Add electronic music to dataset

# Study of the features

We are going to use the following features that Spotify gives provides from their audio analysis:

`duration_ms`	The duration of the track in milliseconds.

`key`	The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation.

`mode` Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.

`time_signature` An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).

`acousticness` A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.

`danceability` Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. 

`energy` Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.

`instrumentalness` Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.

`liveness` Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.

`loudness` The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db. 

`speechiness` Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.

`valence` A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)

`tempo`	The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.

Since our dataset doesnt have any Live song, we assume that:

- liveness won't bei mportant in the prediction

The rest of the features are probably important for the prediction, but we will check that later in the Analysis section.

# Creating new dataset

The procedure to get the features of the tracks will be:
1. Use the track and artist name to get the id 
2. Get all the features by inputing the id of the track

As we start fetching the data, we find out that for some songs:
1. No analysis of features has been done by spotify
2. They simply dont exist in spotify or dont have the same name

Those cases will be ignored

Once we create a dataset unifying artist + track + features + genre, we will check how many tracks do we have left from the initial dataset.
//TODO

# Analysis

Now we proceed with the analysis of the data that we have prepared.
We will first proceed with typical data processing procedures that could improve the model performance.
- Clean data (missing data): we did this before
- Visually explain the data and create assumptions
- Feature selection: do we need all features? 

The study of the features is performed and documented in the script

By feature engineering we want to answer: 

    What is the best representation of the sample data to learn a solution to your problem?

The only normality assumption that regression makes is a conditional one - that conditional on the independent (input) variables, we have that the errors are normally distributed, but in classification we can't measure the errors that way
Every model has its own assumptions and limitations. While there might be a model that can only learn on completely normally distributed data - there are plenty of models which don't, such as SVMs or Random Forests [2] (https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/)

# Prediction 

- Cross validation
- Predict with normalizing data
- Predict selecting all features
- Predict reducing multicollinearity
- Predict with one hot encoding all features
- Predict reducing multicollinearity with one hot encoding
- Ensembling
- Address underfitting and overfitting


We will apply cross validation to:
- Reduce variance
- Avoid overfitting

Prediction is documented in the script

# Metrics

Here we only will use Accuracy since we need to know if the prediction of genre is okay
We dont want to penalise anything. If it was a spam problem classification then we would but we arent doing that

# Conclusion

By checking the accuracy of the prediction, we can see that the maximum one barely reaches the 41.2%

Less than half of the genres are predicted correctly.

We assume one explanation for this: Features very similar between genres, as we saw in the violin plot, so it is difficult to find a difference between them.

We conclude that audio features given by Spotify are not good features to predict the genre of a track.
As a future work, we could try to replicate other project that uses Lyrics to predict the genre of a song.
Also, sound interesting to check also:
- Audio preview features (time series?)
- Album cover
- Popularity

We could even combine them with the features of this project and compare the predictions with the individual projects.
Anyway, this is a task for future work.
Even though, the predictions weren't successful, we learned about ML, DS, project organisation and Python, which I am more than satisfied with this.

Greetings,
Nacho

