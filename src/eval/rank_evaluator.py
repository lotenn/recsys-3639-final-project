from enum import Enum

import pandas as pd
import torch
from torch import no_grad
from torch.utils.data import DataLoader

from src.model.candidate_generation import CandidateGeneration


class RankMetric(str, Enum):
    MRR = "mrr"
    HIT_RATE = "hit_rate"


class RankEvaluator(object):
    def __init__(self, top_k, device):
        self._top_k = top_k
        self._device = device
        self._model = None

    def fit(self, model: CandidateGeneration):
        self._model = model

    def _forward_pass(self, positives, negatives, search, features):
        with no_grad():
            positives = positives.to(self._device)
            negatives = negatives.to(self._device)
            search = search.to(self._device)
            features = features.to(self._device)
            return self._model(positives, negatives, search, features)

    def _find_first_hit(self, labels: pd.Series, candidates: torch.Tensor) -> torch.Tensor:
        """
        Find the first hit in the candidates
        :param labels: torch.Tensor, shape = (batch_size,)
        :param candidates: torch.Tensor, shape = (batch_size, top_k)
        :return:
        """
        first_hit = (candidates[:, :self._top_k] == labels.view(-1, 1)).nonzero()[:, 1]
        return first_hit

    def _calc_metric(self, val_loader: DataLoader, metric: RankMetric) -> float:
        """
        Calculate the metric
        :param val_loader: DataLoader
        :param metric: str, the metric to calculate
        :return:
        """
        score = 0
        for batch in val_loader:
            positives, negatives, search, features, labels = batch
            logits, _ = self._forward_pass(positives, negatives, search, features)
            _, candidates = torch.sort(logits, descending=True, dim=-1)
            score += getattr(self, f"_{metric}")(labels, candidates)
        return score / len(val_loader)

    def _mrr(self, labels: pd.Series, candidates: torch.Tensor) -> float:
        """
        Mean Reciprocal Rank @ top_K
        :param labels: torch.Tensor, shape = (batch_size,)
        :param candidates: torch.Tensor, shape = (batch_size, n_items)
        :return: the MRR score for the given batch
        """
        first_hit = self._find_first_hit(labels, candidates)
        mrr = torch.sum(1.0 / (first_hit + 1)) / labels.shape[0]
        return mrr.item()

    def _hit_rate(self, labels: pd.Series, candidates: torch.Tensor) -> float:
        """
        Hit Ratio @ top_K
        :param labels: torch.Tensor, shape = (batch_size,)
        :param candidates: torch.Tensor, shape = (batch_size, n_items)
        :return: the hit rate score for the given batch
        """
        first_hit = self._find_first_hit(labels, candidates)
        hit_rate = first_hit.shape[0] / labels.shape[0]
        return hit_rate

    def mrr(self, val_loader: DataLoader) -> float:
        """
        Mean Reciprocal Rank @ top_K
        :param val_loader: DataLoader
        :return: the MRR score for the given dataset
        """
        return self._calc_metric(val_loader, RankMetric.MRR)

    def hit_rate(self, val_loader: DataLoader) -> float:
        """
        Hit Ratio @ top_K
        :param val_loader: DataLoader
        :return: the hit rate score for the given dataset
        """
        return self._calc_metric(val_loader, RankMetric.HIT_RATE)
