import pandas as pd
import numpy as np
import operator

def data_to_dict(id1, id2, ratings):
    dict_data = {}

    for _id1_ in sorted(ratings[id1].unique()):
        indices = (ratings[id1] == _id1_)

        dict_data[_id1_] = {}

        for _id2_, rating in zip(ratings[indices][id2], ratings[indices]['rating']):
            dict_data[_id1_][_id2_] = rating.astype('float64')

    return dict_data

def pearson_correlation(dict_data, user1, user2):
    # To get both rated users
    both_rated = {}
    for item in dict_data[user1]:
        if item in dict_data[user2]:
            both_rated[item] = 1

    number_of_ratings = len(both_rated)

    # Checking for number of ratings in common
    if number_of_ratings == 0:
        return 0

    # Add up all the preferences of each item
    user1_preferences_sum = sum([dict_data[user1][item] for item in both_rated])
    user2_preferences_sum = sum([dict_data[user2][item] for item in both_rated])

    # Sum up the squares of preferences of each item
    user1_square_preferences_sum = sum([pow(dict_data[user1][item],2) for item in both_rated])
    user2_square_preferences_sum = sum([pow(dict_data[user2][item],2) for item in both_rated])

    # Sum up the product value of both preferences for each item
    product_sum_of_both_items = sum([dict_data[user1][item] * dict_data[user2][item] for item in both_rated])

    # Calculate the pearson score
    numerator_value = product_sum_of_both_items - (user1_preferences_sum * user2_preferences_sum / number_of_ratings)
    denominator_value = np.sqrt((user1_square_preferences_sum - pow(user1_preferences_sum, 2) / number_of_ratings) \
    * (user2_square_preferences_sum - pow(user2_preferences_sum, 2) / number_of_ratings))

    if denominator_value == 0:
        return 0
    else:
        r = numerator_value/denominator_value
        return r

def calc_corr(dict_data):
    M = len(dict_data)
    corr = np.ones([M + 1, M + 1])

    for i in range(1, M + 1):
        for j in range(i + 1, M + 1):
            corr[i, j] = pearson_correlation(dict_data, i, j)
            corr[j, i] = corr[i, j]

    return corr

def predict(ratings, dict_data, corr, user):
    # Define a dictation for seen movies
    seen_dict = {}

    # Obtain ratings of seen movies rated by the user
    for movie, rating in zip(ratings[(ratings['user_id'] == user)]['movie_id'], ratings[(ratings['user_id'] == user)]['rating']):
        seen_dict[movie] = rating.astype('float64')

    # Define a dictation for unseen movies
    unseen_dict = {}

    # Obtain IDs of unseen movies
    unseen_movies = set(movie_user_data) - set(ratings[(ratings['user_id'] == user)]['movie_id'])

    users_list = set(dict_data.keys())

    # Predict ratings for unseen movies
    for unseen_movie in unseen_movies:
        # Define numerator and denominator
        numer = 0
        denom = 0

        # Calculate numerator and denominator
        for other in users_list:
            if corr[user, other] > 0 and unseen_movie in dict_data[other]:
                numer += corr[user, other] * dict_data[other][unseen_movie]
                denom += corr[user, other]

        # Predict for the uncorrelated item
        if denom == 0:
            unseen_dict[unseen_movie] = np.mean(ratings[(ratings['movie_id'] == unseen_movie)]['rating'].astype('float64'))

        # Predict for the correlated item
        else:
            unseen_dict[unseen_movie] = numer / denom

    return sorted(unseen_dict.items(), key = operator.itemgetter(1))

# Reading users file
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')

# Reading ratings file
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')

# Reading items file
i_cols = ['movie_id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')

# Change data from pandas table to dictionary
movie_user_data = data_to_dict('movie_id', 'user_id', ratings)
user_movie_data = data_to_dict('user_id', 'movie_id', ratings)

# Calculate correlation among items
try:
    corr = np.load('corr.npy')
except:
    corr = calc_corr(user_movie_data)

# Recommend movies by predicting ratings
user_id = 200

recommendation = predict(ratings, user_movie_data, corr, user_id)
toprated = sorted(user_movie_data[user_id].items(), key = operator.itemgetter(1))

recommendation.reverse()
toprated.reverse()

print('Top 10 recommendation movies list for user')
for mov, rat in recommendation[:10]:
    title = items.loc[items['movie_id'] == mov]['movie title'].values
    genre = items.columns.values[np.where((items.loc[items['movie_id'] == mov] == 1).values.squeeze())]
    print(title, genre)

print('\n')
print('Top 10 rated movies list by user')
for mov, rat in toprated[:10]:
    title = items.loc[items['movie_id'] == mov]['movie title'].values
    genre = items.columns.values[np.where((items.loc[items['movie_id'] == mov] == 1).values.squeeze())]
    print(title, genre)
