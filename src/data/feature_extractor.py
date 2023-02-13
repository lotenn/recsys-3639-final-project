import itertools as it
import math

import pandas as pd
from numpy import column_stack, array, hstack
from sklearn import preprocessing
from sklearn.cluster import KMeans
from uszipcode import SearchEngine
from random import sample


class FeatureExtractor:
    def __init__(self, window_size=16, num_negatives=100):
        self.window_size = window_size
        self.num_negatives = num_negatives

        self.genres = None
        self.movies_df = None
        self.users_df = None
        self.ratings_df = None

    def fit_transform(self, genres, movies_df, users_df, ratings_df) -> pd.DataFrame:
        self.genres = genres.copy()
        self.movies_df = movies_df.copy()
        self.users_df = users_df.copy()
        self.ratings_df = ratings_df.copy()

        self._generate_item_features()
        self._generate_user_features()
        return self._generate_examples()

    def _generate_item_features(self) -> pd.DataFrame:
        self.movies_df['genres'] = self.movies_df.apply(lambda x: [i for i, g in enumerate(self.genres) if x[g]],
                                                        axis=1)

    def _gen_f_usr_geo(self, k):
        # Cluster zip codes by latlon to get area feat
        search = SearchEngine()
        latlon = self.users_df['zip_code'].apply(lambda x: search.by_zipcode(x)).apply(
            lambda x: [x.lat or 37.41, x.lng or -122.09] if x else [37.41, -122.09]
        )
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(array(list(latlon.values)))
        self.users_df['f_geo_area'] = kmeans.labels_
        geo_oh = column_stack(pd.get_dummies(self.users_df.f_geo_area).values).T
        print('One-Hot user geo area matrix shape', geo_oh.shape)
        return geo_oh

    def _gen_f_usr_sex(self) -> pd.DataFrame:
        self.users_df['f_sex'] = pd.Categorical(self.users_df.sex).codes
        sex_oh = column_stack(pd.get_dummies(self.users_df.f_sex).values).T
        print('One-Hot user gender matrix shape', sex_oh.shape)
        return sex_oh

    def _gen_f_usr_occu(self) -> pd.DataFrame:
        self.users_df['f_occupation'] = pd.Categorical(self.users_df.occupation).codes
        occupation_oh = column_stack(pd.get_dummies(self.users_df.f_occupation).values).T
        print('One-Hot user occupation matrix shape', occupation_oh.shape)
        return occupation_oh

    def _norm_f_age(self):
        # Normalize ages
        self.users_df['f_age'] = preprocessing.StandardScaler().fit_transform(
            self.users_df.age.values.reshape(-1, 1)
        ).squeeze()
        user_age = array(list(self.users_df.f_age.values)).reshape((-1, 1))
        return user_age

    def _generate_user_features(self, k=4):
        occupation_oh = self._gen_f_usr_occu()
        sex_oh = self._gen_f_usr_sex()
        geo_oh = self._gen_f_usr_geo(k)
        user_age = self._norm_f_age()
        self.users_df['f_vec'] = list(hstack([occupation_oh, sex_oh, geo_oh, user_age]))

    def _gen_f_example_age(self):
        ratings_df_tmp = self.ratings_df.join(self.movies_df, on='movie_id')
        self.ratings_df['example_age'] = (ratings_df_tmp['rating_date'] - ratings_df_tmp['release_date']).apply(
            lambda x: float(x.days))
        self.ratings_df['example_age'] = self.ratings_df['example_age'].fillna(self.ratings_df['example_age'].mean())
        self.ratings_df['example_age'] = preprocessing.StandardScaler().fit_transform(
            self.ratings_df['example_age'].values.reshape(-1, 1)
        ).squeeze()

    def _generate_examples(self) -> pd.DataFrame:
        # Extract Example Age using release_date field
        self._gen_f_example_age()
        windows_df = self._gen_windows_df()
        examples_df = self._gen_examples_df(windows_df)
        print('Examples shape ', examples_df.shape)
        return examples_df

    def _gen_examples_df(self, windows_df):
        # Balance examples
        def balance(df, key, replace=False):
            n_examples = math.ceil(df.groupby(key).size().mean())
            print(f'Using {n_examples} examples per {key}')
            balanced_df = df.groupby(key) \
                .apply(lambda x: x.sample(min(n_examples, len(x) if not replace else n_examples), replace=replace)) \
                .reset_index(drop=True)
            return balanced_df

        examples_unbalanced_df = windows_df[['user_id', 'positives', 'genres', 'next', 'example_age']].join(
            self.users_df['f_vec'], on='user_id').rename(
            columns={'genres': 'search', 'next': 'label', 'f_vec': 'features'})

        # Balance training data by limiting amount of examples per user
        examples_df = balance(examples_unbalanced_df, 'user_id', False)

        # Balance training data by limiting amount of examples per label
        examples_df = balance(examples_df, 'label', True)

        # Add negative examples by sampling from all movies not in the window
        examples_df['negatives'] = self._generate_train_negatives(examples_df)

        return examples_df

    def _gen_windows_df(self):
        user_ratings_grouped = self.ratings_df.join(
            self.movies_df[['genres']], on='movie_id', how='inner').sort_values(
            by='rating_date').groupby(
            ['user_id']).agg(
            {'movie_id': list, 'rating_date': list, 'binary_rating': list, 'genres': list, 'example_age': list}
        )
        # Split user history using windows
        windows_df = pd.DataFrame(
            it.chain(*it.starmap(self._extract_windows, user_ratings_grouped.iterrows())),
            columns=['user_id', 'movie_ids', 'ratings', 'genres', 'positives', 'next', 'example_age']
        )
        return windows_df

    def _extract_windows(self, user_id, item):
        """
        Extract windows from user history. Each window is a tuple of (user_id, movie_ids, ratings, genres, positives, next)
        :param user_id: user id (int)
        :param item: user history (pd.Series)
        """
        for j in range(len(item['movie_id']) - (self.window_size + 1)):
            movie_ids = item['movie_id'][j:j + self.window_size]
            ratings = item['binary_rating'][j:j + self.window_size]
            genres = list(set(it.chain(*item['genres'][j:j + self.window_size])))
            positives = list(array(movie_ids)[array(ratings, dtype=bool)])
            next_id = item['movie_id'][j + self.window_size]
            example_age = item['example_age'][j + self.window_size]

            yield (
                user_id, movie_ids, ratings, genres,
                positives, next_id, example_age
            )

    def _generate_train_negatives(self, examples_df):
        item_pool = set(self.ratings_df.movie_id)
        negative_samples = examples_df['positives'].apply(lambda x: sample(item_pool - set(x), self.num_negatives))
        return negative_samples
