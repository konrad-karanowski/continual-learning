from os import makedirs
from os.path import exists, join
from typing import List, Union, Tuple, Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import math

__all__ = ['ShapeNet', 'ModelNet10', 'ModelNet40', 'ModelNet40_Auto', 'ModelNet40_Manual']


class Normalize:

    def __call__(self, x):
        x -= np.mean(x, axis=0)
        x = x / np.max(np.linalg.norm(x, axis=1))
        return x


class RandomRotation:

    def __call__(self, x):
        phi = np.random.random(1) * 2 * np.pi
        rotation_matrix = np.array([
            [math.cos(phi), -math.sin(phi), 0],
            [math.sin(phi), math.cos(phi), 0],
            [0, 0, 1]
        ])
        return rotation_matrix.dot(x.T).T


class RandomNoise:
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, pointcloud.shape)

        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud


default_transforms = transforms.Compose([
        Normalize(),
        RandomRotation(),
        RandomNoise(),
        transforms.ToTensor()
    ])


test_transforms = transforms.Compose([
        Normalize(),
        transforms.ToTensor()
    ])


class PointCloudDataset(Dataset):
    def __init__(self,
                 sampling,
                 root_dir: str,
                 download_url: Union[str, List[str]],
                 classes: List[str] = None,
                 transform=default_transforms,
                 split: str = 'train'):
        self.sampling = sampling
        self.root_dir = root_dir
        self.transform = default_transforms if split=='train' else test_transforms
        self.split = split.lower()
        self.urls: List[str] = [download_url] if type(download_url) is str else download_url
        self.classes: List[str] = [] if classes is None else classes
        self._map = {}
        for i, c in enumerate(self.classes):
            self._map[c] = i

        self._maybe_download_data()

        dataset = self.load_dataset()
        self.category_to_id = self.get_category_to_id_mapping()
        self.id_to_category = {v: k for k, v in self.category_to_id.items()}
        self.X, self.y = self.get_data(dataset)

        self.n_classes = len(np.unique(self.y))
        self.many_classes = self.n_classes > 1

        self.X_c = np.asarray([])
        self.y_c = np.asarray([])

    def __len__(self):
        return len(self.X)

    # def __getitem__(self, idx):
    #     X, y = self.X[idx].astype(np.float32), self.y[idx].astype(np.float32)
    #     if self.transform:
    #         X = self.transform(X).squeeze()
    #     return X.T, y

    def __get_item(self, idx):
        if isinstance(idx, slice):
            Xs, ys = [], []
            for i in range(idx.start or 0, min(idx.stop or len(self), len(self)), idx.step or 1):
                X, y = self[i]
                Xs.append(X)
                ys.append(y)
            if Xs:
                Xs = torch.stack(Xs) if isinstance(Xs[0], torch.Tensor) else np.stack(Xs)
                ys = torch.stack(ys) if isinstance(ys[0], torch.Tensor) else np.stack(ys)
            return Xs, ys.flatten()
        else:  # Normal index of type int
            if idx < len(self.X):
                X, y = self.X[idx].astype(np.float32), self.y[idx].astype(np.float32)
            else:
                X, y = self.X_c[idx - len(self.X)].astype(np.float32), self.y_c[idx - len(self.X)].astype(np.float32)
                X = np.concatenate((X, X[np.random.choice(len(X), self.sampling - len(X))]))

            if self.transform:
                X = self.transform(X).squeeze()

            return X.T.float(), y

    def __getitem__(self, idx):
        return self.__get_item(idx)

    def get_category_to_id_mapping(self) -> Dict[str, int]:
        if '10' in type(self).__name__.lower():
            return {
                'chair': 0,
                'sofa': 1,
                'bed': 2,
                'monitor': 3,
                'table': 4,
                'toilet': 5,
                'dresser': 6,
                'night_stand': 7,
                'desk': 8,
                'bathtub': 9
            }
        else:
            return {
                'chair': 0,
                'sofa': 1,
                'airplane': 2,
                'bookshelf': 3,
                'bed': 4,
                'vase': 5,
                'monitor': 6,
                'table': 7,
                'toilet': 8,
                'bottle': 9,
                'mantel': 10,
                'tv_stand': 11,
                'plant': 12,
                'piano': 13,
                'desk': 14,
                'dresser': 15,
                'night_stand': 16,
                'car': 17,
                'bench': 18,
                'glass_box': 19,
                'cone': 20,
                'tent': 21,
                'guitar': 22,
                'flower_pot': 23,
                'laptop': 24,
                'keyboard': 25,
                'curtain': 26,
                'sink': 27,
                'lamp': 28,
                'stairs': 29,
                'range_hood': 30,
                'door': 31,
                'bathtub': 32,
                'radio': 33,
                'xbox': 34,
                'stool': 35,
                'person': 36,
                'wardrobe': 37,
                'cup': 38,
                'bowl': 39
            }

    def load_dataset(self) -> Union[np.lib.npyio.NpzFile, h5py.Group]:
        dataset = type(self).__name__.lower()
        if dataset == 'shapenet':
            return np.load(join(self.root_dir, f'{dataset}.npz'))
        elif dataset.startswith('modelnet'):
            split = 'train' if self.split in ['train', 'valid'] else 'test'
            return h5py.File(join(self.root_dir, f'{dataset}.hdf5'), mode='r')[split]

    def add_compressed_data(self, x, y):
        self.X_c = x
        self.y_c = y

    def get_data(self, dataset: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, np.ndarray]:
        chosen_classes = self.classes or dataset.files

        X, y = [], []
        for class_ in chosen_classes:
            n_samples = len(dataset[class_])
            n_train = int(0.85 * n_samples)
            n_valid = int(0.9 * n_samples)

            if self.split == 'train':
                data = dataset[class_][:n_train]
            elif self.split == 'valid':
                data = dataset[class_][n_train:n_valid]
            else:
                data = dataset[class_][n_valid:]

            X.extend(data)
            y.extend(len(data) * [self.category_to_id[class_]])

        X, y = np.array(X, dtype=np.float32), y
        return X, y

    def _maybe_download_data(self):
        if self._data_exists():
            return

        from tqdm import tqdm

        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        print(f'{self.__class__.__name__} doesn\'t exist in root directory `{self.root_dir}`. Downloading...')
        makedirs(self.root_dir, exist_ok=True)

        import urllib.request
        for url in self.urls:
            filename = url.rpartition('/')[2][:-5]
            file_path = join(self.root_dir, filename)

            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
                urllib.request.urlretrieve(url, filename=file_path, reporthook=t.update_to)

    def _data_exists(self) -> bool:
        return (exists(join(self.root_dir, f'{type(self).__name__.lower()}.npz')) or
                exists(join(self.root_dir, f'{type(self).__name__.lower()}.hdf5')))


class ShapeNet(PointCloudDataset):
    def __init__(self,
                 sampling,
                 root_dir: str = '/home/datasets/shapenet',
                 classes: List[str] = None,
                 transform=None,
                 split: str = 'train'):
        super().__init__(
            sampling=sampling,
            root_dir=root_dir,
            download_url='https://www.dropbox.com/s/j7u1ok8q8skgivc/shapenet.npz?dl=1',
            classes=classes,
            transform=transform,
            split=split)

        self.input_size = (3, 2048)


class ModelNet(PointCloudDataset):
    def __init__(self,
                 sampling,
                 root_dir: str,
                 download_url: Union[str, List[str]],
                 classes: List[str] = None,
                 transform=default_transforms,
                 split: str = 'train'):

        assert split.lower() != 'valid', 'Currently only `train` and `test` are supported for MN10'

        super().__init__(
            sampling=sampling,
            root_dir=root_dir,
            download_url=download_url,
            classes=classes,
            transform=transform,
            split=split)

    def get_data(self, dataset):
        chosen_classes = self.classes or list(dataset.keys())

        X, y = [], []
        for class_ in chosen_classes:
            data = dataset[class_]
            to_add = np.asarray([pc[np.random.choice(len(pc), self.sampling, replace=len(pc) < self.sampling)] for pc in data])
            X.extend(to_add)
            y.extend(len(to_add) * [self.category_to_id[class_]])

        return np.array(X), np.array(y)


class ModelNet10(ModelNet):
    def __init__(self,
                 sampling=2048,
                 root_dir: str = '/home/datasets/modelnet10',
                 classes: List[str] = None,
                 transform=default_transforms,
                 split: str = 'train'):
        super().__init__(
            sampling=sampling,
            root_dir=root_dir,
            download_url='https://www.dropbox.com/s/zwx5zejgzewxm9p/modelnet10_15k.hdf5?dl=1',
            classes=classes,
            transform=transform,
            split=split)

    def __len__(self):
        return len(self.X)


class ModelNet40(ModelNet):
    def __init__(self,
                 sampling=2048,
                 root_dir: str = '/data',
                 classes: List[str] = None,
                 transform=default_transforms,
                 split: str = 'train'):
        super().__init__(
            sampling=sampling,
            root_dir=root_dir,
            download_url='https://www.dropbox.com/s/8aniolgxsjzul8g/modelnet40_15k.hdf5?dl=1',
            classes=classes,
            transform=transform,
            split=split)


class ModelNet40_Auto(ModelNet):
    def __init__(self,
                 root_dir: str = '/data',
                 classes: List[str] = None,
                 transform=default_transforms,
                 split: str = 'train'):
        super().__init__(
            sampling=2048,
            root_dir=root_dir,
            download_url='https://www.dropbox.com/s/80qikmb79bi1f9e/modelnet40_auto_15k.hdf5?dl=1',
            classes=classes,
            transform=transform,
            split=split)


class ModelNet40_Manual(ModelNet):
    def __init__(self,
                 sampling,
                 root_dir: str = '/data',
                 classes: List[str] = None,
                 transform=default_transforms,
                 split: str = 'train'):
        super().__init__(
            sampling=sampling,
            root_dir=root_dir,
            download_url='https://www.dropbox.com/s/oqpfgx3o7lw1qmz/modelnet40_manual_15k.hdf5?dl=1',
            classes=classes,
            transform=transform,
            split=split)


class ModelNet40_Stanford(ModelNet):
    def __init__(self,
                 sampling,
                 root_dir: str = '/data',
                 classes: List[str] = None,
                 transform=default_transforms,
                 split: str = 'train'):
        super().__init__(
            sampling=sampling,
            root_dir=root_dir,
            download_url='https://www.dropbox.com/s/ivx5aj4g0xz6pvl/modelnet40_stanford.hdf5?dl=1',
            classes=classes,
            transform=transform,
            split=split)
