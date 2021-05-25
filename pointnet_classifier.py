import torch
from torch import nn


class PointNet(nn.Module):

    def __init__(self, sampling, z_size, batch_norm):
        super().__init__()
        self.name = 'chmurkosieć'

        if batch_norm:
            self._conv1 = nn.Sequential(
                nn.Conv1d(3, 64, 1),
                nn.BatchNorm1d(64),
                nn.ReLU()
            )
        else:
            self._conv1 = nn.Sequential(
                nn.Conv1d(3, 64, 1),
                nn.ReLU()
            )

        if batch_norm:
            self._conv2 = nn.Sequential(
                nn.Conv1d(64, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(),

                nn.Conv1d(128, z_size, 1),
                nn.BatchNorm1d(z_size),
                nn.ReLU()
            )
        else:
            self._conv2 = nn.Sequential(
                nn.Conv1d(64, 128, 1),
                nn.ReLU(),

                nn.Conv1d(128, z_size, 1),
                nn.ReLU()
            )

        self._pool = nn.MaxPool1d(sampling)
        self._flatten = nn.Flatten(1)

    def forward(self, x):
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._pool(x)
        return self._flatten(x)


class PointNetClassifier(nn.Module):

    def __init__(self, sampling, z_size, batch_norm, num_classes):
        super().__init__()
        self.name = 'chmurkosiećklasyfikator'
        self._pn = PointNet(sampling, z_size, batch_norm)
        self.sampling = sampling

        if batch_norm:
            self._dense = nn.Sequential(
                nn.Linear(z_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )
        else:
            self._dense = nn.Sequential(
                nn.Linear(z_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )

        self._final = nn.Linear(256, num_classes)

    def add_final_layer(self, key, num_classes, dropout=0.3):
        self._final[key] = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self._pn(x)
        x = self._dense(x)
        out = self._final(x)
        return out


class NakedPointNetClassifier(nn.Module):

    def __init__(self, z_size, batch_norm, num_classes):
        super().__init__()
        self.name = 'chmurkosiećklasyfikator'

        if batch_norm:
            self._dense = nn.Sequential(
                nn.Linear(z_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )
        else:
            self._dense = nn.Sequential(
                nn.Linear(z_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )

        self._final = nn.Linear(256, num_classes)


    def forward(self, x):
        x = self._dense(x)
        out = self._final(x)
        return out
