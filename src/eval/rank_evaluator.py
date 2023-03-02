from enum import Enum

import pandas as pd
import torch


class RankMetric(str, Enum):
    MRR = "mrr"
    HIT_RATE = "hit_rate"


class RankEvaluator(object):
    def __init__(self, top_k):
        self._top_k = top_k

    def _find_first_hit(self, labels: pd.Series, candidates: torch.Tensor) -> torch.Tensor:
        """
        Find the first hit in the candidates
        :param labels: torch.Tensor, shape = (batch_size,)
        :param candidates: torch.Tensor, shape = (batch_size, top_k)
        :return:
        """
        first_hit = (candidates[:, :self._top_k] == labels.view(-1, 1)).nonzero()[:, 1]
        return first_hit

    def _batch_mrr(self, labels: pd.Series, candidates: torch.Tensor) -> float:
        """
        Mean Reciprocal Rank @ top_K
        :param labels: torch.Tensor, shape = (batch_size,)
        :param candidates: torch.Tensor, shape = (batch_size, n_items)
        :return: the MRR score for the given batch
        """
        first_hit = self._find_first_hit(labels, candidates)
        mrr = torch.sum(1.0 / (first_hit + 1)) / labels.shape[0]
        return mrr.item()

    def _batch_hit_rate(self, labels: pd.Series, candidates: torch.Tensor) -> float:
        """
        Hit Ratio @ top_K
        :param labels: torch.Tensor, shape = (batch_size,)
        :param candidates: torch.Tensor, shape = (batch_size, n_items)
        :return: the hit rate score for the given batch
        """
        first_hit = self._find_first_hit(labels, candidates)
        hit_rate = first_hit.shape[0] / labels.shape[0]
        return hit_rate
