from torch import nn, cat, sort, no_grad
from torch.nn import Embedding
from torch.utils.data import DataLoader

from src.eval.rank_evaluator import RankMetric, RankEvaluator
from src.model.layers import EmbeddingsAggregatorLayer, L2NormLayer


class CandidateGeneration(nn.Module):
    def __init__(self, n_items, n_search_items, n_features, embedding_dim, fc_layers, user_dim):
        super(CandidateGeneration, self).__init__()

        # features layer
        self.positive_embedding = nn.Sequential(
            Embedding(num_embeddings=n_items, embedding_dim=embedding_dim),
            EmbeddingsAggregatorLayer(),
            L2NormLayer()
        )
        self.negative_embedding = nn.Sequential(
            Embedding(num_embeddings=n_items, embedding_dim=embedding_dim),
            EmbeddingsAggregatorLayer(),
            L2NormLayer()
        )
        self.search_embedding = nn.Sequential(
            Embedding(num_embeddings=n_search_items, embedding_dim=embedding_dim),
            EmbeddingsAggregatorLayer(),
            L2NormLayer()
        )

        # FC layers to generate user vector
        layers = [
            nn.Linear(embedding_dim * 3 + n_features, fc_layers[0]),
            nn.ReLU(),
            nn.BatchNorm1d(fc_layers[0]),
        ]

        if len(fc_layers) > 1:
            for in_dim, out_dim in zip(fc_layers[:-1], fc_layers[1:]):
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(out_dim))

        layers.append(nn.Linear(fc_layers[-1], user_dim))
        self.fc = nn.Sequential(*layers)

        self.logits = nn.Sequential(
            nn.Linear(user_dim, n_items),
            nn.Softmax(dim=1)
        )

    def forward(self, positives, negatives, search, features):
        positive_embeddings = self.positive_embedding(positives)
        negative_embeddings = self.negative_embedding(negatives)
        search_embeddings = self.search_embedding(search)

        input_ = cat([positive_embeddings, negative_embeddings, search_embeddings, features], dim=1)
        user = self.fc(input_)
        logits = self.logits(user)

        return logits, user

    def _calc_metric(self, val_loader: DataLoader, rank_eval, metric: RankMetric) -> float:
        """
        Calculate the metric
        :param val_loader: DataLoader
        :param metric: str, the metric to calculate
        :return:
        """
        score = 0
        with no_grad():
            for batch in val_loader:
                positives, negatives, search, features, labels = batch
                logits, _ = self.forward(positives, negatives, search, features)
                _, candidates = sort(logits, descending=True, dim=-1)
                score += getattr(rank_eval, f"_batch_{metric}")(labels, candidates)
        return score / len(val_loader)

    def mrr(self, val_loader: DataLoader, k: int) -> float:
        """
        Mean Reciprocal Rank @ top_K
        :param val_loader: DataLoader
        :param k: int, the top K
        :return: the MRR score for the given dataset
        """
        rank_eval = RankEvaluator(top_k=k)
        return self._calc_metric(val_loader, rank_eval, RankMetric.MRR)

    def hit_rate(self, val_loader: DataLoader, k: int) -> float:
        """
        Hit Ratio @ top_K
        :param val_loader: DataLoader
        :param k: int, the top K
        :return: the hit rate score for the given dataset
        """
        rank_eval = RankEvaluator(top_k=k)
        return self._calc_metric(val_loader, rank_eval, RankMetric.HIT_RATE)
