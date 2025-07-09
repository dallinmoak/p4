import pandas as pd
import numpy as np
from lets_plot import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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
    features,
    labels,
    test_size=0.05,
    random_state=0,
)

gaussianNB_model = GaussianNB()

gaussianNB_model.fit(features_training, labels_training)

gaussianNB_predictions = gaussianNB_model.predict(features_test)

gaussianNB_score = accuracy_score(labels_test, gaussianNB_predictions)

# potential_indicators = [
#     "netprice",
#     "livearea",
#     "totunits",
#     "numbdrm",
#     "numbaths",
#     "nocars",
#     "stories",
# ]

# raw_prospects = df[potential_indicators + ["yrbuilt"]]

# prospects = pd.DataFrame()

# prospects["before_1980"] = raw_prospects["yrbuilt"].apply(lambda x: x < 1980)

# prospects = prospects.join(raw_prospects[potential_indicators])

# before_1980_aggs = prospects.groupby("before_1980").mean().reset_index()

# labels =  taget values (aka answers aka y), don't let the bot know the IDs, but keep it in the same order as the training vectors
# +---------------+
# | 'before_1980' |
# +---------------+
# | FALSE         |
# | TRUE          |


# training_vectors = list all the columns that might predict the target values, with ids, but HIDE the answers (no cheating for the bot)
