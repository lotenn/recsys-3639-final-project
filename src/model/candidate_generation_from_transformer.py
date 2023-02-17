from torch import nn, cat, sort, no_grad
from torch.nn import Embedding, TransformerEncoderLayer, BatchNorm1d
from torch.utils.data import DataLoader

from src.eval.rank_evaluator import RankMetric, RankEvaluator
from src.model.candidate_generation import CandidateGeneration
from src.model.layers import EmbeddingsAggregatorLayer, L2NormLayer, AggMode


class CandidateGenerationT(nn.Module):
    def __init__(self, n_items, n_search_items, n_features, embedding_dim, fc_layers, user_dim):
        super(CandidateGenerationT, self).__init__()

        self.model = CandidateGeneration(n_items, n_search_items, n_features, embedding_dim, fc_layers, user_dim)

        self.model.positive_embedding = nn.Sequential(
            Embedding(num_embeddings=n_items, embedding_dim=embedding_dim),
            TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.1, activation='relu', batch_first=True),
            EmbeddingsAggregatorLayer(),
            L2NormLayer()
        )

        self.model.negative_embedding = nn.Sequential(
            Embedding(num_embeddings=n_items, embedding_dim=embedding_dim),
            TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.1, activation='relu', batch_first=True),
            EmbeddingsAggregatorLayer(),
            L2NormLayer()
        )

        self.model.search_embedding = nn.Sequential(
            Embedding(num_embeddings=n_search_items, embedding_dim=embedding_dim),
            TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.1, activation='relu', batch_first=True),
            EmbeddingsAggregatorLayer(),
            L2NormLayer()
        )

    def forward(self, positives, negatives, search, features):
        return self.model(positives, negatives, search, features)

    def mrr(self, val_loader: DataLoader, k: int, device) -> float:
        """
        Mean Reciprocal Rank @ top_K
        :param device: torch.device
        :param val_loader: DataLoader
        :param k: int, the top K
        :return: the MRR score for the given dataset
        """
        rank_eval = RankEvaluator(top_k=k)
        return self.model._calc_metric(val_loader, rank_eval, RankMetric.MRR, device)

    def hit_rate(self, val_loader: DataLoader, k: int, device) -> float:
        """
        Hit Ratio @ top_K
        :param device: torch.device
        :param val_loader: DataLoader
        :param k: int, the top K
        :return: the hit rate score for the given dataset
        """
        rank_eval = RankEvaluator(top_k=k)
        return self.model._calc_metric(val_loader, rank_eval, RankMetric.HIT_RATE, device)
