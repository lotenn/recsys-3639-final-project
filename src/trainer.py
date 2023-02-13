import time

from torch import no_grad


class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, train_loader, val_loader, epochs=10, verbose=True):
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            start = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss = self._val_epoch(val_loader)
            end = time.time()
            if verbose:
                print(
                    f'Epoch {epoch + 1}/{epochs}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, time: {end - start:.2f}s')
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        return train_losses, val_losses

    def _train_epoch(self, train_loader):
        self.model.train()
        train_loss = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            loss = self.__forward_pass(batch)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader)

    def _val_epoch(self, val_loader):
        self.model.eval()
        val_loss = 0
        with no_grad():
            for batch in val_loader:
                loss = self.__forward_pass(batch)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def __forward_pass(self, batch):
        positives, negatives, search, features, targets = batch
        positives = positives.to(self.device)
        negatives = negatives.to(self.device)
        search = search.to(self.device)
        features = features.to(self.device)
        targets = targets.to(self.device)
        logits, _ = self.model(positives, negatives, search, features)
        loss = self.criterion(logits, targets)
        return loss

