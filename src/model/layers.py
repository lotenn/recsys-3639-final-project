from enum import Enum
from torch import nn, sum, mean


class AggMode(str, Enum):
    MEAN = 'mean'
    SUM = 'sum'


class EmbeddingsAggregatorLayer(nn.Module):
    def __init__(self, agg_mode: AggMode = AggMode.MEAN):
        super(EmbeddingsAggregatorLayer, self).__init__()
        self.agg_mode = agg_mode

    def forward(self, embeddings):
        if self.agg_mode == AggMode.SUM:
            aggregated = sum(embeddings, dim=1)
        elif self.agg_mode == AggMode.MEAN:
            aggregated = mean(embeddings, dim=1)
        return aggregated


class L2NormLayer(nn.Module):

    def forward(self, inputs):
        return nn.functional.normalize(inputs, p=2, dim=1)
