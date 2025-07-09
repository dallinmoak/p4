import pandas as pd
import numpy as np
from lets_plot import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

LetsPlot.setup_html(isolated_frame=True)

df = pd.read_csv("./dwellings_ml.csv")

livearea_frame = df.groupby("before1980")["livearea"].mean().reset_index()

livearea_chart = ggplot(
    data=livearea_frame.reset_index(), mapping=aes(x="before1980", y="livearea")
) + geom_bar(stat="identity")

sprice_frame = df.groupby("before1980")["sprice"].mean().reset_index()

sprice_chart = ggplot(
    data=sprice_frame.reset_index(), mapping=aes(x="before1980", y="sprice")
) + geom_bar(stat="identity")

# i'm excluding parcel 'cause it's not a real meaningful info about the house, and yrbuilt would kind of be cheating.
features = df.drop(columns=["before1980", "yrbuilt", "parcel"])
labels = df["before1980"]

features_training, features_test, labels_training, labels_test = train_test_split(
    features, labels, test_size=0.1, random_state=0
)

gaussianNB_model = GaussianNB()

gaussianNB_model.fit(features_training, labels_training)

gaussianNB_predictions = gaussianNB_model.predict(features_test)

gaussianNB_score = accuracy_score(labels_test, gaussianNB_predictions)

logistic_model = LogisticRegression(max_iter=1000)

logistic_model.fit(features_training, labels_training)

logistic_predictions = logistic_model.predict(features_test)

logistic_score = accuracy_score(labels_test, logistic_predictions)

features_training_scaled = preprocessing.StandardScaler().fit_transform(
    features_training
)
features_test_scaled = preprocessing.StandardScaler().fit_transform(features_test)

logistic_model_scaled = LogisticRegression(max_iter=1000)

logistic_model_scaled.fit(features_training_scaled, labels_training)

logistic_predictions_scaled = logistic_model_scaled.predict(features_test_scaled)

logistic_score_scaled = accuracy_score(labels_test, logistic_predictions_scaled)
