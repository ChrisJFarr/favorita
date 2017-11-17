import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv")
# ['id', 'date', 'store_nbr', 'item_nbr', 'unit_sales', 'onpromotion']
# todo 125M rows in full train, use small subset while exploring
# todo keep stores together when splitting, there are 54 unique store_nbr's
# todo select 5 stores as a subset of train
store_list = np.unique(train.store_nbr)
train = train.loc[[nbr in store_list[0:5] for nbr in train.store_nbr], :].copy()

# EDA Questions for Train set
# todo how do item numbers span across stores?
# todo what are the date ranges? do they vary by store?
# todo how has volume of unit_sales correlated with time?
# todo is the time correlation different by store?
# todo how many items are onpromotion?
# todo how does onpromotion affect a single item_nbr?
# todo how does the volume of the stores compare?
# todo what causal implications does onpromotion have?

# test.csv
test = pd.read_csv("data/test.csv")
test.head()
# ['id', 'date', 'store_nbr', 'item_nbr', 'onpromotion']
# todo onpromotion is boolean in test while 0/1 in train

# sample_submission.csv
sample_submission = pd.read_csv("data/sample_submission.csv")
sample_submission.head()

# holidays_events.csv
holidays_events = pd.read_csv("data/holidays_events.csv")
holidays_events.head()
# ['date', 'type', 'locale', 'locale_name', 'description', 'transferred']
# todo join holidays_events to train/test sets
# todo how do holidays affect volume?
# todo which stores do certain regional holidays affect?
# todo compare locale_name to city and state of stores in the stores.csv for joining holidays

# stores.csv
stores = pd.read_csv("data/stores.csv")
stores.head()
# ['store_nbr', 'city', 'state', 'type', 'cluster']
# todo what does cluster mean?
# todo what does type mean?
# todo are there cities with multiple stores?
# todo: what is the state distribution? city distribution?

# items.csv
items = pd.read_csv("data/items.csv")
items.head()
# ['item_nbr', 'family', 'class', 'perishable']
# todo how many families are there? (family)
# todo is there significance in class for the first digit?
# todo what is the range of classes and how many are there?
# todo what is the distribution of perishable to non?
# todo are certain classes or families more likely to be perishable?
# todo join to train/test using item_nbr

# oil.csv
oil = pd.read_csv("data/oil.csv")
oil.head()
# ['date', 'dcoilwtico']
# todo wtf

# transactions.csv
transactions = pd.read_csv("data/transactions.csv")
transactions.head()
# ['date', 'store_nbr', 'transactions']
# todo which stores are higher/lower volume?
