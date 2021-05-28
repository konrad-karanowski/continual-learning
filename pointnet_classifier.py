from typing import Optional

import torch
from torch import nn
#
#
# class PointNet(nn.Module):
#
#     def __init__(self, sampling, z_size, batch_norm):
#         super().__init__()
#         self.name = 'chmurkosieć'
#
#         if batch_norm:
#             self._conv1 = nn.Sequential(
#                 nn.Conv1d(3, 64, 1),
#                 nn.BatchNorm1d(64),
#                 nn.ReLU()
#             )
#         else:
#             self._conv1 = nn.Sequential(
#                 nn.Conv1d(3, 64, 1),
#                 nn.ReLU()
#             )
#
#         if batch_norm:
#             self._conv2 = nn.Sequential(
#                 nn.Conv1d(64, 128, 1),
#                 nn.BatchNorm1d(128),
#                 nn.ReLU(),
#
#                 nn.Conv1d(128, z_size, 1),
#                 nn.BatchNorm1d(z_size),
#                 nn.ReLU()
#             )
#         else:
#             self._conv2 = nn.Sequential(
#                 nn.Conv1d(64, 128, 1),
#                 nn.ReLU(),
#
#                 nn.Conv1d(128, z_size, 1),
#                 nn.ReLU()
#             )
#
#         self._pool = nn.MaxPool1d(sampling)
#         self._flatten = nn.Flatten(1)
#
#     def forward(self, x):
#         x = self._conv1(x)
#         x = self._conv2(x)
#         x = self._pool(x)
#         return self._flatten(x)
#
#
# class PointNetClassifier(nn.Module):
#
#     def __init__(self, sampling, z_size, batch_norm, num_classes):
#         super().__init__()
#         self.name = 'chmurkosiećklasyfikator'
#         self._pn = PointNet(sampling, z_size, batch_norm)
#         self.sampling = sampling
#
#         if batch_norm:
#             self._dense = nn.Sequential(
#                 nn.Linear(z_size, 512),
#                 nn.BatchNorm1d(512),
#                 nn.ReLU(),
#
#                 nn.Linear(512, 256),
#                 nn.BatchNorm1d(256),
#                 nn.ReLU()
#             )
#         else:
#             self._dense = nn.Sequential(
#                 nn.Linear(z_size, 512),
#                 nn.BatchNorm1d(512),
#                 nn.ReLU(),
#
#                 nn.Linear(512, 256),
#                 nn.BatchNorm1d(256),
#                 nn.ReLU()
#             )
#
#         self._final = nn.Linear(256, num_classes)
#
#     def add_final_layer(self, key, num_classes, dropout=0.3):
#         self._final[key] = nn.Sequential(
#             nn.Linear(256, num_classes),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         x = self._pn(x)
#         x = self._dense(x)
#         out = self._final(x)
#         return out
#
#
# class NakedPointNetClassifier(nn.Module):
#
#     def __init__(self, z_size, batch_norm, num_classes):
#         super().__init__()
#         self.name = 'chmurkosiećklasyfikator'
#
#         if batch_norm:
#             self._dense = nn.Sequential(
#                 nn.Linear(z_size, 512),
#                 nn.BatchNorm1d(512),
#                 nn.ReLU(),
#
#                 nn.Linear(512, 256),
#                 nn.BatchNorm1d(256),
#                 nn.ReLU()
#             )
#         else:
#             self._dense = nn.Sequential(
#                 nn.Linear(z_size, 512),
#                 nn.BatchNorm1d(512),
#                 nn.ReLU(),
#
#                 nn.Linear(512, 256),
#                 nn.BatchNorm1d(256),
#                 nn.ReLU()
#             )
#
#         self._final = nn.Linear(256, num_classes)
#
#
#     def forward(self, x):
#         x = self._dense(x)
#         out = self._final(x)
#         return out


class PointNetRepresentationModel(nn.Module):
    def __init__(self, z_size: int = 1024, encoder_last_nonlinearity: str = 'identity', use_tnets: bool = True,
                 no_batch_norm: bool = False, add_fc: bool = False, add_reparametrization: bool = False):
        super().__init__()
        self.name = 'chmurkosieć2!!!'

        self.use_tnets = use_tnets
        self.add_fc = add_fc
        self.add_reparametrization = add_reparametrization

        if self.use_tnets:
            self.tnet3 = TNet(k=3)

        if no_batch_norm:
            self.PN1 = nn.Sequential(
                nn.Conv1d(3, 64, 1),
                nn.ReLU(inplace=True),

                nn.Conv1d(64, 128, 1),
                nn.ReLU(inplace=True)
            )
        else:
            self.PN1 = nn.Sequential(
                nn.Conv1d(3, 64, 1),
                nn.BatchNorm1d(64, momentum=0.5),
                nn.ReLU(inplace=True),

                nn.Conv1d(64, 64, 1),
                nn.BatchNorm1d(64, momentum=0.5),
                nn.ReLU(inplace=True)
            )

        if self.use_tnets:
            self.tnet64 = TNet(k=64)

        if no_batch_norm:
            self.PN2 = nn.Sequential(
                nn.Conv1d(64, 64, 1),
                nn.ReLU(inplace=True),

                nn.Conv1d(64, 128, 1),
                nn.ReLU(inplace=True),

                nn.Conv1d(128, z_size, 1),
            )
        else:
            self.PN2 = nn.Sequential(
                nn.Conv1d(64, 64, 1),
                nn.BatchNorm1d(64, momentum=0.5),
                nn.ReLU(inplace=True),

                nn.Conv1d(64, 128, 1),
                nn.BatchNorm1d(128, momentum=0.5),
                nn.ReLU(inplace=True),

                nn.Conv1d(128, z_size, 1),
                nn.BatchNorm1d(z_size, momentum=0.5),
            )

        # Activation after PN2, but before maxpooling
        activations = {'relu': nn.ReLU(),
                       'tanh': nn.Tanh(),
                       'identity': nn.Identity()}
        self.last_activation = activations[encoder_last_nonlinearity]

        if add_fc:
            self.fc = nn.Sequential(
                nn.Linear(z_size, z_size),
                nn.BatchNorm1d(z_size, affine=False)
            )

        if add_reparametrization:
            self.mu_layer = nn.Linear(z_size, z_size, bias=True)
            self.logvar_layer = nn.Linear(z_size, z_size, bias=True)

        self.init_weights()

    def init_weights(self):
        nonlinearity = 'relu'

        def weights_init(m: nn.Module):
            classname = m.__class__.__name__
            if classname in ('Conv1d', 'Linear'):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            if classname.startswith('BatchNorm'):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.PN1.apply(weights_init)
        self.PN2.apply(weights_init)

        if self.add_fc:
            self.fc.apply(weights_init)
        if self.add_reparametrization:
            self.mu_layer.apply(weights_init)
            self.logvar_layer.apply(weights_init)

    # noinspection PyUnboundLocalVariable
    def forward(self, x):
        if self.use_tnets:
            input_transform = self.tnet3(x)

            # Nx3*(1+n_n)x2048 = (Nx2048x3*(1+n_n) @ Nx3*(1+n_n)x3*(1+n_n))^T
            x = torch.bmm(x.transpose(1, 2), input_transform).transpose(1, 2)

        # Nx64x2048
        features = self.PN1(x)

        if self.use_tnets:
            # Nx64x64
            feature_transform = self.tnet64(features)
            # Nx64x2048 = (Nx2048x64 @ Nx64x64)^T
            features = torch.bmm(features.transpose(1, 2), feature_transform).transpose(1, 2)

        # NxZ_SIZEx2048
        point_features = self.last_activation(self.PN2(features))
        global_features = torch.max(point_features, dim=2).values

        if self.add_fc:
            global_features = torch.relu(self.fc(global_features))

        if self.add_reparametrization:
            mu = self.mu_layer(global_features)
            logvar = self.logvar_layer(global_features)
            global_features = self.reparameterize(mu, logvar)

        if self.use_tnets:
            if self.add_reparametrization:
                return input_transform, feature_transform, mu, logvar, global_features
            else:
                return input_transform, feature_transform, global_features
        else:
            if self.add_reparametrization:
                return mu, logvar, global_features
            else:
                return global_features

    def forward_with_projection(self, x):
        out = self(x)

        if self.use_tnets:
            projection = self.fc(out[-2])
        else:
            projection = self.fc(out)

        return out, projection

    @staticmethod
    def reparameterize(mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def compress(self, x):
        x_rotated = x.detach().clone()

        if self.use_tnets:
            input_transform = self.tnet3(x)

            # Nx3*(1+n_n)x2048 = (Nx2048x3*(1+n_n) @ Nx3*(1+n_n)x3*(1+n_n))^T
            x_rotated = torch.bmm(x_rotated.transpose(1, 2), input_transform).transpose(1, 2)

        # Nx64x2048
        features = self.PN1(x_rotated)

        if self.use_tnets:
            # Nx64x64
            feature_transform = self.tnet64(features)
            # Nx64x2048 = (Nx2048x64 @ Nx64x64)^T
            features = torch.bmm(features.transpose(1, 2), feature_transform).transpose(1, 2)

        # NxZ_SIZEx2048
        point_features = self.last_activation(self.PN2(features))

        point_indices = torch.argmax(point_features, dim=2)

        return torch.stack([x_.T[pi] for x_, pi in zip(x, point_indices)])


class PointNetClassificationModel(nn.Module):
    def __init__(self, num_classes: int, z_size: int = 1024, encoder_last_nonlinearity: str = 'relu',
                 use_tnets: bool = True, no_batch_norm: bool = False, dropout: Optional[float] = None,
                 add_fc: bool = True, add_reparametrization: bool = True, pointnet = None):
        super().__init__()
        self.name = 'chmurkasiećklasyfikator2'

        self.pointnet = pointnet

        # TODO Check with MLP
        if dropout:
            self.PN3 = nn.Sequential(
                nn.Linear(z_size, 512),
                nn.BatchNorm1d(512, momentum=0.5),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256, momentum=0.5),
                nn.Dropout(p=dropout),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_classes),
            )
        else:
            self.PN3 = nn.Sequential(
                nn.Linear(z_size, 512),
                nn.BatchNorm1d(512, momentum=0.5),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256, momentum=0.5),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_classes),
            )

    def forward(self, pc):
        if self.pointnet.use_tnets:
            if self.pointnet.add_reparametrization:
                input_transform, feature_transform, mu, logvar, global_features = self.pointnet(pc)
                y_ = self.PN3(global_features)
                return input_transform, feature_transform, mu, logvar, global_features, y_
            else:
                input_transform, feature_transform, global_features = self.pointnet(pc)
                y_ = self.PN3(global_features)
                return input_transform, feature_transform, global_features, y_
        else:
            if self.pointnet.add_reparametrization:
                mu, logvar, global_features = self.pointnet(pc)
                y_ = self.PN3(global_features)
                return mu, logvar, global_features, y_
            else:
                global_features = self.pointnet(pc)
                y_ = self.PN3(global_features)
                return global_features, y_

    def compress(self, x):
        return self.pointnet.compress(x)

    def encode(self, x):
        return self.pointnet(x)

    def update_bn_momentum(self, momentum):
        def _momentum_update(m: nn.Module):
            if m.__class__.__name__.startswith('BatchNorm'):
                m.momentum = momentum

        self.pointnet.PN1.apply(_momentum_update)
        self.pointnet.PN2.apply(_momentum_update)
        if self.pointnet.use_tnets:
            self.pointnet.tnet3.apply(_momentum_update)
            self.pointnet.tnet64.apply(_momentum_update)
        self.PN3.apply(_momentum_update)


class TNet(nn.Module):

    def __init__(self, k):
        super().__init__()

        self.k = k

        self.conv_part = nn.Sequential(
            nn.Conv1d(self.k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.fc_part = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, self.k * self.k)
        )

        self.init_weights()

    def init_weights(self):
        nonlinearity = 'relu'

        def weights_init(m: nn.Module):
            classname = m.__class__.__name__
            if classname in ('Conv1d', 'Linear'):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        nonlinearity = 'relu'
        self.conv_part.apply(weights_init)
        self.fc_part.apply(weights_init)

        torch.nn.init.constant_(self.fc_part[-1].weight, 0)
        # torch.nn.init.constant_(self.fc_part[-1].bias, 0)
        self.fc_part[-1].bias = nn.Parameter(torch.eye(self.k, device=self.fc_part[-1].bias.device,
                                             dtype=self.fc_part[-1].bias.dtype).flatten())


    def forward(self, pc):
        step_1 = self.conv_part(pc).max(dim=2).values
        step_2 = self.fc_part(step_1)
        return step_2.view(-1, self.k, self.k)