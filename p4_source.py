import pandas as pd
import numpy as np
from lets_plot import *

LetsPlot.setup_html(isolated_frame=True)

df = pd.read_csv('./dwellings_ml.csv')

potential_indicators = [
    'netprice',
    'livearea',
    'totunits',
    'numbdrm',
    'numbaths',
    'nocars',
    'stories',
]

raw_prospects = df[
    potential_indicators + ['yrbuilt']
]

prospects = pd.DataFrame()

prospects['before_1980'] = raw_prospects['yrbuilt'].apply(lambda x: x < 1980)

prospects = prospects.join(raw_prospects[potential_indicators])

before_1980_aggs = prospects.groupby('before_1980').mean().reset_index()

