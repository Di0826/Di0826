import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch

rating_columns = ['user_id', 'item id', 'rating', 'timestamp']
ratings = pd.read_table('ml-100k\\u.data', header=None, names=rating_columns)
num_ratings = ratings.shape[0]

item_columns = ['movie_id', ' movie title', ' release date', ' video release date',
                'IMDb URL', ' unknown', ' Action', ' Adventure', ' Animation',
                'Children', ' Comedy', ' Crime', ' Documentary', ' Drama', ' Fantasy',
                'Film-Noir', ' Horror', ' Musical', ' Mystery', ' Romance', ' Sci-Fi',
                'Thriller', ' War', ' Western']
items = pd.read_table('ml-100k\\u.item', sep='|', header=None, names=item_columns)
num_items = items.shape[0]

user_columns = ['user id', 'age', 'gender', 'occupation', 'zip code']
users = pd.read_table('ml-100k\\u.user', sep='|',  header=None,
                      names=user_columns)
num_users = users.shape[0]

train = pd.read_table('ml-100k\\u1.base', header=None, names=rating_columns)
test = pd.read_table('ml-100k\\u1.test', header=None, names=rating_columns)

items = items.drop(['movie_id', ' movie title', ' release date', ' video release date', 'IMDb URL'], axis=1)
users = users.drop(['user id', 'occupation', 'zip code'], axis=1)
users = users.replace('M', 0).replace('F', 1)

print(items.head())
print(users.head())




