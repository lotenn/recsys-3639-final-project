from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch import LongTensor, FloatTensor


class RatingsDataset(Dataset):
    """Wrapper, convert <'positives', 'negatives', 'search', 'features', 'label'> Tensor into Pytorch Dataset"""

    def __init__(self, positives_tensor, negatives_tensor, search_tensor, features_tensor, labels_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for each user-item pair
        """
        self.positives_tensor = positives_tensor
        self.negatives_tensor = negatives_tensor
        self.search_tensor = search_tensor
        self.features_tensor = features_tensor
        self.labels_tensor = labels_tensor

    def __getitem__(self, index):
        return self.positives_tensor[index], self.negatives_tensor[index], self.search_tensor[index], \
                  self.features_tensor[index], self.labels_tensor[index]

    def __len__(self):
        return self.labels_tensor.size(0)

    @staticmethod
    def from_df(df) -> 'RatingsDataset':
        """
        Create a RatingsDataset from a pandas dataframe
        :param df: a Pandas dataframe with columns 'positives', 'negatives', 'search', 'features', 'label'
        :return:
        """
        return RatingsDataset(
            positives_tensor=pad_sequence(list(map(LongTensor, df['positives'].values)), batch_first=True),
            negatives_tensor=pad_sequence(list(map(LongTensor, df['negatives'].values)), batch_first=True),
            search_tensor=pad_sequence(list(map(LongTensor, df['search'].values)), batch_first=True),
            features_tensor=pad_sequence(list(map(FloatTensor, df['features'].values)), batch_first=True),
            labels_tensor=LongTensor(df['label'].values.tolist())
        )



