from typing import List, Tuple, Any

import seaborn as sns

from matplotlib import pyplot as plt
from pandas import DataFrame
from torch.types import Device
from torch.utils.data import DataLoader


class Plotter:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize

    def plot_loss_curves(self, train_losses, val_losses):
        _, ax = plt.subplots(1, figsize=self.figsize)
        sns.lineplot(data=DataFrame({'train': train_losses, 'val': val_losses}), ax=ax)
        ax.set(title='Train/Validation loss by epoch', xlabel='epoch', ylabel='loss')

    def plot_genres_distribution(self, movies_df):
        plt.figure(figsize=self.figsize)
        movies_df.iloc[:, 5:].sum().sort_values(ascending=False).plot.bar()
        plt.xticks(rotation=45)
        plt.title('Genres distribution')
        plt.show()

    def plot_user_distributions(self, users_df, x='sex', y='age', count='occupation'):
        figsize = (self.figsize[0] * 2, self.figsize[1])
        _, ax = plt.subplots(1, 2, figsize=figsize)
        sns.violinplot(data=users_df, y=y, ax=ax[0], x=x)
        sns.countplot(users_df[count], ax=ax[1], order=users_df[count].value_counts().index)
        plt.xticks(rotation=45)
        ax[0].set(title=f'{y} distribution among users in the dataset, split by {x}')
        ax[1].set(title=f'{count} count distribution among users in the dataset')

    def plot_ratings_distribution(self, ratings_df):
        ratings_per_user = ratings_df.groupby('user_id')['rating'].count()
        figsize = (self.figsize[0] * 2, self.figsize[1])
        _, ax = plt.subplots(1, 2, figsize=figsize)
        sns.histplot(ratings_per_user, ax=ax[0], bins=20)
        sns.boxplot(ratings_per_user, ax=ax[1], showfliers=False)
        ax[0].set(xlabel='number of rating', title='Total number of ratings per user')
        ax[1].set(xlabel='number of rating per user', title='Total number of ratings per user');

    @staticmethod
    def plot_rating_trends(ratings_df):
        df = ratings_df.copy()
        df['rating_month'] = df['rating_date'].dt.to_period('M')
        ratings_count = df.groupby('rating_month')['rating'].count()
        ratings_count.plot()
        plt.title('rating count distribution over time in month granulrity')

    @staticmethod
    def plot_movies_age_distribution(movies_df):
        df = movies_df.copy()
        df['release_year'] = df['release_date'].dt.to_period('Y')
        items_count = df.groupby('release_year')['title'].count()
        items_count.plot()
        plt.title('movies release_date distribution in yearly granulrity')

    def plot_models_performance(self, models: List[Tuple[Any, str]], val_loader: DataLoader, device: Device):
        results_df = DataFrame(columns=['model', 'metric', 'k', 'score'])
        for m, name in models:
            for k in [5, 10, 20]:
                results_df.loc[results_df.shape[0]] = (name, 'mrr', k, m.mrr(val_loader, k, device=device))
                results_df.loc[results_df.shape[0]] = (name, 'hit_rate', k, m.hit_rate(val_loader, k, device=device))

        figsize = (self.figsize[0] * 2, self.figsize[1])
        _, ax = plt.subplots(1, 2, figsize=figsize)
        for i, metric in enumerate(['mrr', 'hit_rate']):
            data = results_df.loc[results_df.metric == metric].drop(columns=['metric'])
            sns.barplot(data=data.sort_values(by='score'), y='score', x='k', hue='model', ax=ax[i])
            ax[i].set(title=f'{metric} validation score')