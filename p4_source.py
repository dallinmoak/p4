import pandas as pd
import numpy as np
from lets_plot import *

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
