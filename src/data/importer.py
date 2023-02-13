from os import path
from typing import Tuple, List

import pandas as pd
from pandas import read_csv, to_datetime
from requests import get
from zipfile import ZipFile


class DataImporter:
    def __init__(self, url, zip_name='ml-100k.zip'):
        self.url = url
        self.zip_name = zip_name

    def _download_data(self):
        r = get(self.url, allow_redirects=True)
        with open(self.zip_name, 'wb') as zip_ref:
            zip_ref.write(r.content)

    def _unzip_data(self):
        with ZipFile(self.zip_name, "r") as zip_ref:
            zip_ref.extractall()

    def _get_users_df(self) -> pd.DataFrame:
        users_df = read_csv(
            'ml-100k/u.user',
            sep='|',
            names=['user_id', 'age', 'sex', 'occupation', 'zip_code']
        )
        users_df["user_id"] = users_df["user_id"] - 1
        users_df.set_index('user_id', inplace=True)
        return users_df

    def _get_movies_df(self) -> Tuple[pd.DataFrame, List[str]]:
        genres = [
            "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
        ]
        movies_df = read_csv(
            'ml-100k/u.item',
            sep='|',
            header=None,
            names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', *genres],
            encoding="ISO-8859-1"
        )

        movies_df["movie_id"] = movies_df["movie_id"] - 1
        movies_df["release_date"] = to_datetime(movies_df['release_date'], format='%d-%b-%Y')
        movies_df.drop(columns=['video_release_date', 'imdb_url'], inplace=True)
        movies_df.set_index('movie_id', inplace=True)
        return movies_df, genres

    def _get_ratings_df(self) -> pd.DataFrame:
        dtypes = {'user_id': int, 'movie_id': int, 'rating': float, 'unix_timestamp': int}

        ratings_df = read_csv(
            'ml-100k/u.data',
            sep='\t',
            names=dtypes.keys(),
            dtype=dtypes
        )

        ratings_df["user_id"] = ratings_df["user_id"] - 1
        ratings_df["movie_id"] = ratings_df["movie_id"] - 1
        ratings_df['rating_date'] = to_datetime(ratings_df['unix_timestamp'], unit='s')
        ratings_df.drop(columns=['unix_timestamp'], inplace=True)
        ratings_df['binary_rating'] = 1
        return ratings_df

    def import_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not path.exists('ml-100k'):
            self._download_data()
            self._unzip_data()
        else:
            print('Data already exists')

        users_df = self._get_users_df()
        movies_df, genres = self._get_movies_df()
        ratings_df = self._get_ratings_df()
        return users_df, movies_df, genres, ratings_df
