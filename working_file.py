# todo create simple prediction model and make submission

import pandas as pd
import numpy as np
import pickle
from datetime import date
from dateutil.relativedelta import relativedelta

# Train
train = pd.read_csv("data/train.csv")
train.loc[:, "date"] = pd.to_datetime(train.date)
store_list = np.unique(train.store_nbr)
# # Test
# test = pd.read_csv("data/test.csv")
# test.loc[:, "date"] = pd.to_datetime(test.date)
# # Sample Submission
# sample_submission = pd.read_csv("data/sample_submission.csv")

# Build subset of train and split new test
train = train.loc[[nbr in store_list[0:5] for nbr in train.store_nbr], :].copy()  # this is slow, run sparingly
# Test is the last two weeks of the train split todo split this multiple times when validating
train.head()
# train time range
test_stop = max(train.date)
test_start = test_stop + relativedelta(weeks=-2)
train = train.set_index("date").loc[:test_start, :].reset_index()
test = train.set_index("date").loc[test_start:test_stop, :].reset_index()

# Simple prediction model components
# regression
# time series: lag variables, trend components, seasonality
# additional features: onpromotion
# final features:
# a few lag variables
# todo create a lag dataset
# match store_nbr and item_nbr and date
# date is the perspective date, allowing all features to be relative to this date and contain the same columns

date(2017, 8, 13).weekday()

# split out one item, for one location, then use .shift() for lags


# todo one lag variable should be average of the same day of the week 2-x number of weeks back
# todo could use a recurrent neural network to capture the trend of lag variables
# add day of week feature to df
train.loc[:, "weekday"] = [dt.weekday() for dt in train.date]
test.loc[:, "weekday"] = [dt.weekday() for dt in test.date]
# Add time to index for easy filtering
train.index = train.date
test.index = test.date
# Calculate averages using group by
train.head()
# Calculate averages through a sliding range window
# todo: lag will only work if there aren't missing values, fill in missing values in sequence when item not sold
train = train.set_index(["date", "store_nbr", "item_nbr"]).unstack(level=["store_nbr", "item_nbr"]).stack(level=["store_nbr", "item_nbr"], dropna=False)
train.loc[:, "unit_sales"] = train.loc[:, "unit_sales"].fillna(0)
# todo vars: using shift and groupby, create lag variables per week from 2 months + 2 weeks back to 2 weeks back
train.loc[:, "shift1"] = train.groupby(["store_nbr", "item_nbr", "weekday"])["unit_sales"].shift(1)

start = 6  # Since it's by weekday, each lag shift(1) is a full week back
stop = 2  # to 2 weeks back
for i in range(stop, start):
    train.loc[:, "shift_%s" % str(i)] = train.groupby(["store_nbr", "item_nbr", "weekday"])["unit_sales"].shift(i)

# todo start here: after adding the lag variables, after a visual inspection do any look problematic?

# How do I complete this in a dynamic way? This only calculates for the test set
weeks_prior_2 = train.loc[test_start + relativedelta(days=-7):test_start, :].groupby(["store_nbr", "item_nbr", "weekday"])["unit_sales"].mean()

# How does it need to be structured to use shift? (use with group by?)


itm = 103665
store = 1
weekday = 0
train.loc[(train.item_nbr == itm) & (train.store_nbr == store) & (train.weekday == weekday), ["unit_sales", "shift1"]]




# one month back, average each week day for given item_nbr and store
# filter to month range
lag_df = train.loc[(train.item_nbr == 103665) & (train.store_nbr == 1), ]  # filter to store and item combo
lag_df.set_index("date", inplace=True)  # move date to index
# filter to date > -2 month < perspective date -2 weeks
# from the date in test, start 2 weeks prior and stop a month from that
perspective_date = date(2017, 8, 16)  # this is the date of the sales volume being predicted
end_date = perspective_date + relativedelta(weeks=-2)  # the last sales records for the example
start_date = end_date + relativedelta(months=-2)  # the first sales records for the example

lag_df.loc[start_date:end_date,:]

# Robust model ideas
# additional features: onpromotion, regional holiday boolean, oil price




