import pandas as pd
import numpy as np

def read_user():
    user_columns = ['user id', 'age', 'gender', 'occupation', 'zip code']
    users = pd.read_table('ml-100k\\u.user', sep='|', header=None,
                          names=user_columns)
    users = users.drop(['user id', 'occupation', 'zip code'], axis=1)
    users = users.replace('M', 0).replace('F', 1)
    users = np.asarray(users)
    return users


def read_item():
    item_columns = ['movie_id', 'movie title', 'release date', 'video release date',
                    'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                    'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                    'Thriller', 'War', 'Western']
    items = pd.read_table('ml-100k\\u.item', sep='|', header=None, names=item_columns)
    items['day'], items['month'], items['year'] = items['release date'].str.split('-').str
    items = items.drop(['movie_id', 'movie title', 'release date', 'video release date', 'IMDb URL'], axis=1)
    items = items.drop(['day', 'month'], axis=1)
    items['year'] = items.fillna('1970')
    items['year'] = items['year'].astype('int64')
    items = np.asarray(items)
    return items


def read_rating_train(train_file_num: int):
    rating_columns = ['user_id', 'item id', 'rating', 'timestamp']
    train = pd.read_table('ml-100k\\u' + str(train_file_num) + '.base', header=None, names=rating_columns)
    train = np.asarray(train)
    train[:, 0] = train[:, 0] - 1
    train[:, 1] = train[:, 1] - 1
    return train


def read_rating_test(train_file_num: int):
    rating_columns = ['user_id', 'item id', 'rating', 'timestamp']
    test = pd.read_table('ml-100k\\u' + str(train_file_num) + '.test', header=None, names=rating_columns)
    test = np.asarray(test)
    test[:, 0] = test[:, 0] - 1
    test[:, 1] = test[:, 1] - 1

    return test
