# todo create simple prediction model and make submission

import pandas as pd
import numpy as np
import pickle
from datetime import date
from dateutil.relativedelta import relativedelta

# # Train
# train = pd.read_csv("data/train.csv")
# train.loc[:, "date"] = pd.to_datetime(train.date)
# store_list = np.unique(train.store_nbr)
# # Test
# test = pd.read_csv("data/test.csv")
# test.loc[:, "date"] = pd.to_datetime(test.date)
# # Sample Submission
# sample_submission = pd.read_csv("data/sample_submission.csv")

# # Build subset of train and split new test
# train = train.loc[[nbr in store_list[0:5] for nbr in train.store_nbr], :].copy()  # this is slow, run sparingly
# train.to_csv("data/sub_train.csv")

# # train time range
# test_stop = max(train.date)
# test_start = test_stop + relativedelta(weeks=-2)
# train = train.set_index("date").loc[:test_start, :].reset_index()
# test = train.set_index("date").loc[test_start:test_stop, :].reset_index()

# Pull in subsetted dataset for exploring models
train = pd.read_csv("data/sub_train.csv", index_col=0)

# Simple prediction model components
# same day of week lag variables
# 2 weeks prior: shift(2)
# 3 weeks prior: shift(3)
# 2 weeks prior to 2 weeks + 1 month prior average: for i in range(2, 6): shift(i)

# Creating lag dataset
# groupby(["store_nbr", "item_nbr", "weekday"])
# Use shift(1) to go one week back
# weekday() counts from sunday (0) to saturday (6)

# Modeling
# todo could use a recurrent neural network to capture the trend of lag variables
# todo start with simple models: linear regression, tree

# Pre-prep
train.loc[:, "date"] = pd.to_datetime(train.date)


# todo: lag will only work if there aren't missing values, fill in missing values in sequence when item not sold
# todo: for test set will need to combine train and test prior to stack/unstack
# todo after splitting train/test, remove items missing id in test set
train = train.set_index(["date", "store_nbr", "item_nbr"]).unstack(
    level=["store_nbr", "item_nbr"]).stack(level=["store_nbr", "item_nbr"], dropna=False)  # Fill missing rows
train.loc[:, "unit_sales"] = train.loc[:, "unit_sales"].fillna(0)  # Replace nan with 0's
train.reset_index(inplace=True)
train.loc[:, "weekday"] = [dt.weekday() for dt in train.date]  # add day of week feature to df
# train.index = train.date  # Add time to index for easy filtering
start = 6  # Since it's by weekday, each lag shift(1) is a full week back
stop = 2  # to 2 weeks back
for i in range(stop, start):
    # todo using shift and groupby, create lag variables
    train.loc[:, "shift_%s" % str(i)] = train.groupby(["store_nbr", "item_nbr", "weekday"])["unit_sales"].shift(i)

# todo store processed sub train for future use
train.to_csv("data/processed_sub_train.csv")
train = pd.read_csv("data/processed_sub_train.csv", index_col=0)

# todo clean NaN's from shifting, be sure to not drop NaN's from id
train = train.dropna(subset=["shift_%s" % str(i) for i in range(stop, start)])

# todo add additional feature for month average
# train.loc[:, "prior_mnth"] = train.apply(lambda x: np.mean([x["shift_%s" % str(i)] for i in range(stop, start)]), axis=1)

# No need to create test/train split while testing... cross validation using lags
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_log_error, make_scorer
model = LinearRegression()
x = train.loc[:, ["shift_%s" % str(i) for i in range(stop, start)]]
x = x.iloc[0:100, :]
x.fillna(0, inplace=True)
y = train.loc[:, "unit_sales"]
y = y[0:100]
y[y < 0] = 0  # Replace returns with 0 for now
y.fillna(0, inplace=True)
max(y)
score = cross_val_score(model, x, y, scoring=make_scorer(mean_squared_log_error), cv=10, n_jobs=3)
pred = cross_val_predict(model, x, y, cv=10, n_jobs=3)
np.mean(np.sqrt(score))

pd.DataFrame({"actual": y, "pred": pred})
# todo start here!!: note: approach seems to work, make first submission

# todo start here: after adding the lag variables, after a visual inspection do any look problematic?
train.head()

itm = 103665
store = 1
weekday = 0
train.loc[(train.item_nbr == itm) & (train.store_nbr == store) & (train.weekday == weekday), ["unit_sales", "shift_2"]]

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
