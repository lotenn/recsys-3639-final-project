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
        fc_layers = list(fc_layers)
        fc_layers.insert(0, embedding_dim * 3 + n_features)
        layers = []
        if len(fc_layers) > 1:
            for in_dim, out_dim in zip(fc_layers[:-1], fc_layers[1:]):
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU()
                ])

        layers.append(nn.Linear(fc_layers[-1], user_dim))
        self.fc = nn.Sequential(*layers)

        self.output = nn.Sequential(
            nn.Linear(user_dim, n_items),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, positives, negatives, search, features):
        positive_embeddings = self.positive_embedding(positives)
        negative_embeddings = self.negative_embedding(negatives)
        search_embeddings = self.search_embedding(search)

        input_ = cat([positive_embeddings, negative_embeddings, search_embeddings, features], dim=1)
        user = self.fc(input_)
        predictions = self.output(user)

        return predictions, user

    def _calc_metric(self, val_loader: DataLoader, rank_eval, metric: RankMetric, device) -> float:
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
                positives = positives.to(device)
                negatives = negatives.to(device)
                search = search.to(device)
                features = features.to(device)

                logits, _ = self.forward(positives, negatives, search, features)
                _, candidates = sort(logits, descending=True, dim=-1)

                labels, candidates = labels.to('cpu'), candidates.to('cpu')

                score += getattr(rank_eval, f"_batch_{metric}")(labels, candidates)
        return score / len(val_loader)

    def mrr(self, val_loader: DataLoader, k: int, device) -> float:
        """
        Mean Reciprocal Rank @ top_K
        :param device: torch.device
        :param val_loader: DataLoader
        :param k: int, the top K
        :return: the MRR score for the given dataset
        """
        rank_eval = RankEvaluator(top_k=k)
        return self._calc_metric(val_loader, rank_eval, RankMetric.MRR, device)

    def hit_rate(self, val_loader: DataLoader, k: int, device) -> float:
        """
        Hit Ratio @ top_K
        :param device: torch.device
        :param val_loader: DataLoader
        :param k: int, the top K
        :return: the hit rate score for the given dataset
        """
        rank_eval = RankEvaluator(top_k=k)
        return self._calc_metric(val_loader, rank_eval, RankMetric.HIT_RATE, device)
