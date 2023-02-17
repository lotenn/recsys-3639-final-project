import pandas as pd
from torch import LongTensor
from torch.utils.data import DataLoader

from src.eval.rank_evaluator import RankMetric, RankEvaluator


class Popularity(object):
    def __init__(self):
        self._ratings = None
        self._candidates = None

    def fit(self, ratings: pd.DataFrame):
        self._ratings = ratings
        self._candidates = LongTensor(self._ratings['movie_id'].value_counts().sort_values().index)

    def predict(self, batch, k=None):
        return self._candidates.tile(batch.shape[0], 1)[:, :k]

    def _calc_metric(self, val_loader: DataLoader, rank_eval, metric: RankMetric) -> float:
        """
        Calculate the metric
        :param val_loader: DataLoader
        :param metric: str, the metric to calculate
        :return:
        """
        score = 0
        for batch in val_loader:
            positives, negatives, search, features, labels = batch
            candidates = self.predict(positives)
            score += getattr(rank_eval, f"_batch_{metric}")(labels, candidates)
        return score / len(val_loader)

    def mrr(self, val_loader: DataLoader, k: int, **kwargs) -> float:
        """
        Mean Reciprocal Rank @ top_K
        :param val_loader: DataLoader
        :param k: int, the top K
        :return: the MRR score for the given datasets
        """
        rank_eval = RankEvaluator(top_k=k)
        return self._calc_metric(val_loader, rank_eval, RankMetric.MRR)

    def hit_rate(self, val_loader: DataLoader, k: int, **kwargs) -> float:
        """
        Hit Ratio @ top_K
        :param val_loader: DataLoader
        :param k: int, the top K
        :return: the hit rate score for the given dataset
        """
        rank_eval = RankEvaluator(top_k=k)
        return self._calc_metric(val_loader, rank_eval, RankMetric.HIT_RATE)
