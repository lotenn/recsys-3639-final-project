from torch import nn
from torch.nn import Embedding, TransformerEncoderLayer

from src.model.candidate_generation import CandidateGeneration
from src.model.layers import EmbeddingsAggregatorLayer, L2NormLayer


class CandidateGenerationT(CandidateGeneration):
    def __init__(self, n_items, n_search_items, n_features, embedding_dim, fc_layers, user_dim):
        super(CandidateGenerationT, self).__init__(n_items, n_search_items, n_features, embedding_dim, fc_layers, user_dim)

        self.positive_embedding = nn.Sequential(
            Embedding(num_embeddings=n_items, embedding_dim=embedding_dim),
            TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.1, activation='relu', batch_first=True),
            EmbeddingsAggregatorLayer(),
            L2NormLayer()
        )

        self.negative_embedding = nn.Sequential(
            Embedding(num_embeddings=n_items, embedding_dim=embedding_dim),
            TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.1, activation='relu', batch_first=True),
            EmbeddingsAggregatorLayer(),
            L2NormLayer()
        )

        self.search_embedding = nn.Sequential(
            Embedding(num_embeddings=n_search_items, embedding_dim=embedding_dim),
            TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.1, activation='relu', batch_first=True),
            EmbeddingsAggregatorLayer(),
            L2NormLayer()
        )
