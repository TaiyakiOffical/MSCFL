import logging
import math
import random
import copy
import torch
from torch.autograd import Variable
from torchvision.transforms import transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import numpy as np
import torch.utils.data as data

from dataset import *
from miscellaneous import print2log, get_args

min_require_size = 100


def load_Purchase100(datadir):

    data = np.load(datadir + '/purchase100.npz')
    features = data['features']
    labels = data['labels']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
    y_test = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)


    return X_train, y_train, X_test, y_test

def load_Texas100(datadir):

    data = np.load(datadir + "/texas100.npz")
    features = data['features']
    labels = data['labels']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
    y_test = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

    return X_train, y_train, X_test, y_test

def load_mnist_data(datadir):

    # transform只有在以[]方式获取数据时生效
    # transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_dataset = MNIST_truncated(datadir, train=True, download=True)
    mnist_test_dataset = MNIST_truncated(datadir, train=False, download=True)

    data_train, targets_train = mnist_train_dataset.data, mnist_train_dataset.targets
    data_test, targets_test = mnist_test_dataset.data, mnist_test_dataset.targets

    # data_train = data_train.numpy()
    # targets_train = targets_train.numpy()
    # data_test = data_test.numpy()
    # targets_test = targets_test.numpy()

    return data_train, targets_train, data_test, targets_test

def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.targets
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.targets

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_fmnist_data(datadir):

    # transform = transforms.Compose([transforms.ToTensor()])

    fmnist_train_dataset = FMNIST_truncated(datadir, train=True, download=True)
    fmnist_test_dataset = FMNIST_truncated(datadir, train=False, download=True)

    data_train, targets_train = fmnist_train_dataset.data, fmnist_train_dataset.targets
    data_test, targets_test = fmnist_test_dataset.data, fmnist_test_dataset.targets

    # print(data_train.shape, targets_train.shape, data_test.shape, targets_test.shape)

    return data_train, targets_train, data_test, targets_test


def load_femnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FEMNIST(datadir, train=True, transform=transform, download=True)
    mnist_test_ds = FEMNIST(datadir, train=False, transform=transform, download=True)

    data_train, targets_train, user_train = mnist_train_ds.data, mnist_train_ds.targets, mnist_train_ds.users_index
    data_test, targets_test, user_test = mnist_test_ds.data, mnist_test_ds.targets, mnist_test_ds.users_index

    # X_train = X_train.data.numpy()
    # y_train = y_train.data.numpy()
    # u_train = np.array(u_train)
    # X_test = X_test.data.numpy()
    # y_test = y_test.data.numpy()
    # u_test = np.array(u_test)

    user_train = np.array(user_train)
    user_test = np.array(user_test)

    return (data_train, targets_train, user_train, data_test, targets_test, user_test)


def load_svhn_data(datadir):

    # transform = transforms.Compose([transforms.ToTensor()])

    svhn_train_dataset = SVHN_truncated(datadir, train=True, download=True)
    svhn_test_dataset = SVHN_truncated(datadir, train=False, download=True)

    data_train, targets_train = svhn_train_dataset.data, svhn_train_dataset.targets
    data_test, targets_test = svhn_test_dataset.data, svhn_test_dataset.targets

    # print(data_train.shape, targets_train.shape, data_test.shape, targets_test.shape)
    # print(len(targets_test))
    # print(len(targets_train))

    return data_train, targets_train, data_test, targets_test


def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_dataset = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_dataset = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    data_train, targets_train = cifar10_train_dataset.data, cifar10_train_dataset.targets
    data_test, targets_test = cifar10_test_dataset.data, cifar10_test_dataset.targets

    # test_img, test_target = cifar10_train_ds[500]
    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return data_train, targets_train, data_test, targets_test


def iid_partition(targets, n_parties):
    if isinstance(targets, np.ndarray):
        N = targets.shape[0]
    else:
        N = len(targets)
    data_indexes = np.random.permutation(N)
    # print(data_indexes)
    index_batch = np.array_split(data_indexes, n_parties)
    parties_data = {party: index_batch[party] for party in range(n_parties)}

    return parties_data

# 标签偏移
def quantity_based_LDS_partition(targets, n_parties, target_kinds, alpha = 1, alpha2 = 0):
    targets = np.array(targets)
    parties_data_indexes = {party: np.array([]) for party in range(n_parties)}
    unique_target_kinds = np.unique(targets)
    original_indices = np.arange(len(targets))
    combined = list(zip(targets, original_indices))
    np.random.shuffle(combined)
    shuffled_targets, shuffled_indices = zip(*combined)
    # 将结果转换回 numpy 数组
    shuffled_targets = np.array(shuffled_targets)
    shuffled_indices = np.array(shuffled_indices)
    split_point = int(alpha * len(targets))
    targets1 = shuffled_targets[:split_point]
    targets1_indices = shuffled_indices[:split_point]
    targets2 = shuffled_targets[split_point:]
    targets2_indices = shuffled_indices[split_point:]
    cluster_target_kinds = np.array_split(unique_target_kinds, math.ceil(len(unique_target_kinds) / target_kinds))
    cluster_parties = np.array_split(np.array(range(n_parties)), len(cluster_target_kinds))
    index_batch_targets2 = np.array_split(targets2_indices, n_parties)
    for labels, parties in zip(cluster_target_kinds, cluster_parties):
        print2log('---->quantity_based_LDS_partition results: parties: %s, labels: %s' % (parties, labels))
        for label in labels:
            # 获取标签索引
            label_indexes = []
            label_indexes_temp = np.where(targets1 == label)[0]
            for index in label_indexes_temp:
                label_indexes.append(targets1_indices[index])
            label_indexes = np.array(label_indexes)
            # 蔟内iid划分数据
            index_batch = np.array_split(label_indexes, len(parties)) 
            for index, party in enumerate(parties):
                parties_data_indexes[party] = np.append(parties_data_indexes[party], index_batch[index])
    if alpha != 1:
        for party in range(n_parties):
            parties_data_indexes[party] = np.append(parties_data_indexes[party], index_batch_targets2[party])
    if alpha2 != 0:
        num_to_remove = int(len(parties_data_indexes[0]) * alpha2)
        num_to_add = num_to_remove
        remove_indices = np.random.choice(len(parties_data_indexes[0]), num_to_remove, replace=False)
        parties_data_indexes[0] = np.delete(parties_data_indexes[0], remove_indices)
        add_indices = np.random.choice(len(parties_data_indexes[2]), num_to_add, replace=False)
        data_to_add = parties_data_indexes[2][add_indices]
        parties_data_indexes[0] = np.concatenate((parties_data_indexes[0], data_to_add))
    return parties_data_indexes


def dist_based_LDS_partition(targets, n_parties, beta):
    # 防止party数据量过少
    min_size = 0
    # train的数据总数
    N = targets.shape[0]
    # target_kinds代表标签种类数量
    target_kinds = np.unique(targets).shape[0]
    # np.random.seed(2020)

    index_batch = [[]]
    while min_size < min_require_size:
        # [[], [], [], [], [], [], [], [], [], []]二维数组
        index_batch = [[] for _ in range(n_parties)]
        for k in range(target_kinds):
            # print("k:", k)
            # 标签为K的数据的索引
            indexes_k = np.where(targets == k)[0]
            np.random.shuffle(indexes_k)
            # 标签种类为K的所有数据在所有参与方的分布比例
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            # Balance
            proportions = np.array([p * (len(idx_b) < N / n_parties) for p, idx_b in zip(proportions, index_batch)])
            proportions = proportions / proportions.sum()
            # print("proportion3: ", proportions)
            proportions = (np.cumsum(proportions) * len(indexes_k)).astype(int)[:-1]
            # print("proportion4: ", proportions)
            index_batch = [idx_b + idx.tolist() for idx_b, idx in zip(index_batch, np.split(indexes_k, proportions))]
            min_size = min([len(idx_b) for idx_b in index_batch])

    parties_data = {}
    for j in range(n_parties):
        # 因为每个idx_bach数据都是从标签1开始生成具有一定顺序，所以需要打乱顺序
        np.random.shuffle(index_batch[j])
        parties_data[j] = index_batch[j]

    return parties_data


class GaussianNoise(object):
    def __init__(self, mean=0, std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class SwapLabel(object):
    def __init__(self, first_label=0, second_label=0):
        self.first_label = first_label
        self.second_label = second_label

    def __call__(self, label):
        temp = label
        if label == self.first_label:
            label = self.second_label
        elif label == self.second_label:
            label = self.first_label

        if isinstance(label, int):
            # print('former_label: %s, new_label: %s' % (temp, label))
            label = torch.tensor(label)
        return label

    def __repr__(self):
        return self.__class__.__name__ + '(first_label=%s, second_label=%s)' % (self.first_label, self.second_label)


def quantity_skew_partition(targets, n_parties, beta):
    N = targets.shape[0]
    data_indexes = np.random.permutation(N)
    min_size = 0

    proportions = []
    while min_size < 10:
        proportions = np.random.dirichlet(np.repeat(beta, n_parties))
        proportions = proportions / proportions.sum()
        min_size = np.min(proportions * N)
    proportions = (np.cumsum(proportions) * N).astype(int)[:-1]
    index_batch = np.split(data_indexes, proportions)
    parties_data = {i: index_batch[i] for i in range(n_parties)}

    return parties_data

def real_partition(n_parties, user_index):
    num_user = user_index.shape[0]
    user = np.zeros(num_user+1,dtype=np.int32)
    for i in range(1,num_user+1):
        user[i] = user[i-1] + user_index[i-1]
    no = np.random.permutation(num_user)
    batch_idxs = np.array_split(no, n_parties)
    parties_data = {i:np.zeros(0,dtype=np.int32) for i in range(n_parties)}
    for i in range(n_parties):
        for j in batch_idxs[i]:
            parties_data[i]=np.append(parties_data[i], np.arange(user[j], user[j+1]))
    return parties_data

def get_dataloader(dataset, datadir, train_batch_size, test_batch_size, train_indexes, test_indexes,
                   noise=0, swapped_label=False, rotate_angle=0):
    # transfrom自动在遍历dataloader时生效
    target_transform_train = None
    target_transform_test = None
    if rotate_angle != 0:
        transform_train = transforms.Compose([
            transforms.RandomRotation([rotate_angle, rotate_angle]) if dataset not in ["texas100", "purchase100"] else transforms.Lambda(lambda x: x),
            transforms.Lambda(lambda x: torch.flip(x, dims=[0])) if dataset in ["texas100", "purchase100"] else transforms.Lambda(lambda x: x),
            GaussianNoise(0, noise),
        ])
        transform_test = transforms.Compose([
            transforms.RandomRotation([rotate_angle, rotate_angle]) if dataset not in ["texas100", "purchase100"] else transforms.Lambda(lambda x: x),
            transforms.Lambda(lambda x: torch.flip(x, dims=[0])) if dataset in ["texas100", "purchase100"] else transforms.Lambda(lambda x: x),
            transforms.ToTensor() if dataset not in ["texas100", "purchase100"] else transforms.Lambda(lambda x: x)] )
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor() if dataset not in ["texas100", "purchase100"] else transforms.Lambda(lambda x: x),
            GaussianNoise(0, noise),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor() if dataset not in ["texas100", "purchase100"] else transforms.Lambda(lambda x: x),
        ])
    if swapped_label == True:
        target_transform_train = transforms.Compose([
            SwapLabel(i, 9-i) for i in range(5)
        ])
        target_transform_test = transforms.Compose([
            SwapLabel(i, 9-i) for i in range(5)
        ])

    if dataset == 'mnist':
        dataset_obj = MNIST_truncated
    elif dataset == 'fmnist':
        dataset_obj = FMNIST_truncated
    elif dataset == 'femnist':
        dataset_obj = FEMNIST
    elif dataset == 'svhn':
        dataset_obj = SVHN_truncated
    elif dataset == "purchase100":
        dataset_obj = Purchase100
    elif dataset == "texas100":
        dataset_obj = Texas100
    elif dataset == 'cifar10':
        dataset_obj = CIFAR10_truncated
        if rotate_angle != 0:
            transform_train = transforms.Compose([
                
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),

                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation([rotate_angle, rotate_angle]),

                transforms.ToTensor(),
                transforms.Lambda(lambda img: torch.rot90(img, k=2, dims=(1, 2))),

                GaussianNoise(0, noise)
            ])
            transform_test = transforms.Compose([
                # transforms.RandomRotation([rotate_angle, rotate_angle]),
                transforms.ToTensor(),
                transforms.Lambda(lambda img: torch.rot90(img, k=2, dims=(1, 2))),
                GaussianNoise(0, noise)
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                GaussianNoise(0, noise)
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                GaussianNoise(0, noise)
            ])
    elif dataset == 'cifar100':
            dataset_obj = CIFAR100_truncated

            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            if rotate_angle != 0:
                transform_train = transforms.Compose([
                    # transforms.ToPILImage(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda img: torch.rot90(img, k=2, dims=(1, 2))),
                    normalize
                ])
                # data prep for test set
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda img: torch.rot90(img, k=2, dims=(1, 2))),
                    normalize])
            else:
                transform_train = transforms.Compose([
                    # transforms.ToPILImage(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
                # data prep for test set
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    normalize])
    else:
        print2log("Dataset Not Supported!!!")
        exit(-1)
    dataset_train = dataset_obj(datadir, data_indexes=train_indexes, train=True, transform=transform_train, target_transform=target_transform_train)
    dataset_test = dataset_obj(datadir, data_indexes=test_indexes, train=False, transform=transform_test, target_transform=target_transform_test)
    # if rotate_angle != 0:
    #     rotation_transform = transforms.RandomRotation(degrees=(rotate_angle-0.1,rotate_angle+0.1))
    #     for i in range(len(dataset_train)):
    #         img_train, _ = dataset_train[i]
    #         dataset_train[i] = rotation_transform(img_train)
    #     for i in range(len(dataset_test)):
    #         img_test,_ = dataset_test[i]
    #         dataset_test[i] = rotation_transform(img_test)
    train_dataloader = data.DataLoader(dataset=dataset_train, batch_size=train_batch_size, shuffle=True, drop_last=False)
    test_dataloader = data.DataLoader(dataset=dataset_test, batch_size=test_batch_size, shuffle=False, drop_last=False)

    return train_dataloader, test_dataloader


def partition_data(dataset, datadir, partition, n_parties, train_batch_size ,test_batch_size=32, noise=0, seed=0,
                   malicious=(), poisoning_type=None, target_kinds=None, beta=None, alpha = 1, alpha2 = 0):
    # 设置随机种子
    np.random.seed(seed)
    # 读取数据
    print2log('Getting raw data...')
    data_train, targets_train, data_test, targets_test = [], [], [], []
    if dataset == 'mnist':
        data_train, targets_train, data_test, targets_test = load_mnist_data(datadir)
    elif dataset == 'fmnist':
        data_train, targets_train, data_test, targets_test = load_fmnist_data(datadir)
    elif dataset == 'svhn':
        data_train, targets_train, data_test, targets_test = load_svhn_data(datadir)
    elif dataset == 'cifar10':
        data_train, targets_train, data_test, targets_test = load_cifar10_data(datadir)
    elif dataset == 'femnist':
        data_train, targets_train, user_train, data_test, targets_test, user_test = load_femnist_data(datadir)  
    elif dataset == "cifar100":
        data_train, targets_train, data_test, targets_test = load_cifar100_data(datadir)
    elif dataset == "texas100":
        data_train, targets_train, data_test, targets_test = load_Texas100(datadir)
    elif dataset == "purchase100":
        data_train, targets_train, data_test, targets_test = load_Purchase100(datadir)

    else:
        print2log("Dataset Not Supported!!!")
        exit(-1)
    print2log('Getting raw data succeeded!')

    # print(data_train.shape)
    # exit(-1)

    # 拆分数据
    print2log('Partitioning data...')
    parties_train_data = {}
    parties_test_data = {}
    if partition in ['iid', "feature_imbalance_divide_four", "feature_imbalance_divide_two","swap_label"]:
        parties_train_data = iid_partition(targets_train, n_parties)
        parties_test_data = iid_partition(targets_test, n_parties)
    elif partition == 'quantity_based_LDS':
        parties_train_data = quantity_based_LDS_partition(targets_train, n_parties, target_kinds, alpha, alpha2)
        parties_test_data = quantity_based_LDS_partition(targets_test, n_parties, target_kinds, alpha, alpha2)
    elif partition == 'dist_based_LDS':
        parties_train_data = dist_based_LDS_partition(targets_train, n_parties, beta)
        parties_test_data = dist_based_LDS_partition(targets_test, n_parties, beta)
    elif partition == 'synthetic_feature_imbalance' or partition == 'real-world_feature_imbalance':
        if dataset != 'femnist':
            print2log("only femnist dataset can suit this")
            exit(-1)
        parties_train_data = real_partition(n_parties, user_train)
        parties_test_data = real_partition(n_parties, user_test)
    elif partition == 'quantity_skew_partition':
        parties_train_data = quantity_skew_partition(targets_train, n_parties, beta)
        parties_test_data = quantity_skew_partition(targets_test, n_parties, beta)
    else:
        print2log("Specified Partition Method Not Supported!!!")
        exit(-1)
    print2log('Partitioning data succeeded!')

    # 读取party各自的dataloader
    print2log('Getting dataloader...')
    party_dataloader_train = {}
    party_dataloader_test = {}
    for i in range(n_parties):
        swapped_label1, swapped_label2 = 0, 0
        if i in malicious and poisoning_type == 'label-swapped':
            while swapped_label1 == swapped_label2:
                swapped_label1 = random.randint(0, 9)
                swapped_label2 = random.randint(0, 9)
        if partition == "feature_imbalance_divide_four":
            party_dataloader_train[i], party_dataloader_test[i] = \
                get_dataloader(dataset, datadir, train_batch_size, test_batch_size, parties_train_data[i], parties_test_data[i],
                            noise=noise, swapped_label=False,rotate_angle=(i*90 % 360))
        elif partition == "feature_imbalance_divide_two":
            party_dataloader_train[i], party_dataloader_test[i] = \
                get_dataloader(dataset, datadir, train_batch_size, test_batch_size, parties_train_data[i], parties_test_data[i],
                            noise=noise, swapped_label=False,rotate_angle=(i*180 % 360))
        elif partition == "swap_label":
            party_dataloader_train[i], party_dataloader_test[i] = \
                get_dataloader(dataset, datadir, train_batch_size, test_batch_size, parties_train_data[i], parties_test_data[i],
                            noise=noise, swapped_label=bool(i%2))
        else:
            party_dataloader_train[i], party_dataloader_test[i] = \
                get_dataloader(dataset, datadir, train_batch_size, test_batch_size, parties_train_data[i], parties_test_data[i],
                            noise=noise)

    print2log('Getting dataloader succeeded!')

    # 获得每个party的训练集大小
    party_dataSize_dict = {}
    for p, d in parties_train_data.items():
        party_dataSize_dict[p] = len(d)

    return party_dataloader_train, party_dataloader_test, party_dataSize_dict


if __name__ == '__main__':
    args = get_args()
